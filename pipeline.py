"""
pipeline.py — Main VLA data collection pipeline
================================================
Orchestrates all modules:
  camera.py        → fisheye undistortion
  imu.py           → IMU parsing + Madgwick fusion
  vts_parser.py    → .vts sync file parsing
  kinematics.py    → 6-DoF math, grasp, joint angles, dynamics
  segmentation.py  → IMU-guided step timeline
  compositor.py    → 4-panel annotated video
  export.py        → CSV + JSONL

Usage (single video):
    python pipeline.py \\
        --input recording.mp4 \\
        --imu   recording.imu \\
        --vts   recording.vts \\
        --macro_task "Disassemble a car wheel" \\
        --steps "Attach socket;Loosen nuts;Remove wheel" \\
        --nl_caption "Operator disassembles a car wheel with an impact wrench." \\
        --environment "Car Workshop" --scene "Car service" \\
        --op_height 162 --robot_height 120 --skip 2

Usage (batch folder):
    python pipeline.py --batch /path/to/dataset_root/ [same flags]

Folder structure for batch mode:
    dataset_root/
        clip001.mp4
        clip001.imu
        clip001.vts    (optional)
        clip002.mp4
        ...

Output files are placed next to each input video unless --output_dir is set.
"""

from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, PoseLandmarkerOptions
import urllib.request

from camera      import FisheyeUndistorter
from imu         import IMUStream
from vts_parser  import VTSParser
from kinematics  import (palm_frame, rpy_from_R, world_hand_rpy, world_landmarks,
                          normalize_pose, finger_joint_angles, classify_grasp,
                          SmoothedDynamics, hw, hn, pw, _cs as cs_scale)
from segmentation import StepTimeline
from compositor  import build_frame
from export      import (CSVWriter, JSONLWriter,
                          build_csv_row, build_jsonl_record, build_hand_json)


# ── MediaPipe model URLs ──────────────────────────────────────────────────────

HAND_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
POSE_URL  = ("https://storage.googleapis.com/mediapipe-models/"
             "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")


def _dl_model(url: str, path: str) -> str:
    if not os.path.exists(path):
        print(f"  Downloading {os.path.basename(path)} …")
        urllib.request.urlretrieve(url, path)
        print("  Done.")
    return path


# ── Single video processor ────────────────────────────────────────────────────

def process_video(
    input_path:    str | Path,
    imu_path:      str | Path | None = None,
    vts_path:      str | Path | None = None,
    csv_path:      str | Path        = "hand_6dof.csv",
    jsonl_path:    str | Path        = "vla_dataset.jsonl",
    out_path:      str | Path        = "vla_output.mp4",
    out_w:         int               = 1600,
    out_h:         int               = 720,
    skip:          int               = 1,
    max_hands:     int               = 2,
    conf:          float             = 0.55,
    macro_task:    str               = "",
    steps:         list[str]         = None,
    nl_caption:    str               = "",
    ts_start:      float             = 0.0,
    ts_end:        float             = 0.0,
    environment:   str               = "Factory",
    scene:         str               = "Workstation",
    op_height:     float             = 170.0,
    robot_height:  float             = 0.0,
    write_video:   bool              = True,
    # Camera calibration overrides (None = use defaults from spec sheet)
    cam_K:         np.ndarray | None = None,
    cam_D:         np.ndarray | None = None,
    cam_balance:   float             = 0.5,
    # IMU options
    imu_beta:      float             = 0.033,
    imu_mag:       bool              = True,
    # Dynamics window
    dyn_window:    int               = 9,
) -> dict:

    input_path = Path(input_path)
    print(f"\n{'='*60}")
    print(f"  Video : {input_path.name}")

    # ── Stage 1: Camera undistorter ───────────────────────────────────────────
    undistorter = FisheyeUndistorter(K=cam_K, D=cam_D, balance=cam_balance)
    print(f"  Camera undistorter ready  (balance={cam_balance})")

    # ── Stage 1b: VTS ────────────────────────────────────────────────────────
    vts = VTSParser(vts_path)
    print(f"  VTS : {vts}")

    # ── Stage 1c: IMU ─────────────────────────────────────────────────────────
    imu_available = False
    imu_stream    = None
    if imu_path and Path(imu_path).exists():
        imu_stream = IMUStream(imu_path, beta=imu_beta, mag_enabled=imu_mag)
        print(f"  IMU loaded: {imu_stream}")
        print(f"  Running Madgwick fusion …")
        imu_stream.fuse()
        print(f"  Fusion complete.")
        imu_available = True
    else:
        print("  IMU file not provided / not found — orientation will be camera-relative only.")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = max(1.0, fps / skip)
    if ts_end <= ts_start:
        ts_end = total / fps

    # ── Video writer ──────────────────────────────────────────────────────────
    writer = None
    if write_video:
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps, (out_w, out_h),
        )

    # ── MediaPipe detectors ───────────────────────────────────────────────────
    hand_model = _dl_model(HAND_URL, "hand_landmarker.task")
    pose_model = _dl_model(POSE_URL, "pose_landmarker_lite.task")

    hand_opts = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_opts = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Step timeline ─────────────────────────────────────────────────────────
    steps_list = steps or []
    timeline   = StepTimeline(
        steps    = steps_list,
        ts_start = ts_start,
        ts_end   = ts_end,
        imu_stream = imu_stream if imu_available else None,
    )
    print(f"  Step timeline: {timeline}")

    # ── Per-hand smoothed dynamics state ─────────────────────────────────────
    dynamics: dict[str, SmoothedDynamics] = {}

    # ── IMU accel history for sparkline ──────────────────────────────────────
    imu_accel_hist: deque[float] = deque(maxlen=200)

    # ── Main loop ─────────────────────────────────────────────────────────────
    with (CSVWriter(csv_path) as csv_w,
          JSONLWriter(jsonl_path) as jsonl_w,
          mp_vision.HandLandmarker.create_from_options(hand_opts) as h_det,
          mp_vision.PoseLandmarker.create_from_options(pose_opts) as p_det):

        fidx = 0
        pbar = tqdm(total=total, unit="frame",
                    desc=f"  Processing {input_path.name}")

        while True:
            ret, frame = cap.read()
            if not ret: break
            pbar.update(1)

            if fidx % skip != 0:
                fidx += 1
                continue

            tc    = fidx / fps
            ts_ms = int(tc * 1000)

            # Undistort frame
            undistorted = undistorter.undistort(frame)

            # IMU state at this frame timestamp
            if imu_available:
                # Try VTS for a more precise timestamp
                frame_ts = vts.frame_ts(fidx, fps)
                imu_state = imu_stream.at(frame_ts)
                i_roll, i_pitch, i_yaw = imu_state.euler_deg()
                imu_q = imu_state.quaternion
                cam_accel_world = imu_state.accel_world
                cam_gyro = np.array([imu_state.gx, imu_state.gy, imu_state.gz])
                imu_accel_hist.append(float(np.linalg.norm(cam_accel_world)))
            else:
                from imu import IMUSample
                imu_state = IMUSample(ts=tc,
                    ax=0,ay=0,az=0,gx=0,gy=0,gz=0,mx=0,my=0,mz=0)
                i_roll = i_pitch = i_yaw = 0.0
                imu_q = np.array([1.,0.,0.,0.])
                cam_accel_world = np.zeros(3)
                cam_gyro = np.zeros(3)

            # MediaPipe on undistorted frame
            rgb    = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            h_res  = h_det.detect_for_video(mp_img, ts_ms)
            p_res  = p_det.detect_for_video(mp_img, ts_ms)

            # Step
            micro_step, step_idx, total_steps = timeline.at(tc)

            # ── Per-hand processing ───────────────────────────────────────────
            nlms: list        = []
            hness: list[str]  = []
            cur_hands: list   = []
            haw: dict         = {}
            rpy_data:  dict   = {}
            grasp_data: dict  = {}
            joint_data: dict  = {}
            frame_hands_json  = []

            if h_res.hand_world_landmarks:
                for slot, (wlm, nlm, hi) in enumerate(zip(
                        h_res.hand_world_landmarks,
                        h_res.hand_landmarks,
                        h_res.handedness)):
                    if slot >= max_hands: break

                    label  = hi[0].display_name
                    lms_w  = hw(wlm)          # (21,3) camera-frame, Y-up
                    lms_sc = cs_scale(lms_w)  # scaled for 3D display

                    nlms.append(hn(nlm))
                    hness.append(label)
                    cur_hands.append((lms_sc, label))
                    haw[label] = lms_sc

                    # Camera-relative 6-DoF
                    _, palm_R = palm_frame(lms_w)
                    roll_c, pitch_c, yaw_c = rpy_from_R(palm_R)

                    # World-frame 6-DoF (IMU-fused)
                    roll_w, pitch_w, yaw_w = world_hand_rpy(palm_R, imu_q)
                    rpy_data[label] = (roll_w, pitch_w, yaw_w)

                    # World-frame landmarks
                    lms_world = world_landmarks(lms_w, imu_q)

                    # Height-normalised landmarks
                    lms_norm = normalize_pose(lms_w, op_height, robot_height)

                    # End-effector (wrist in camera frame)
                    ee_xyz = lms_w[0]

                    # Finger joint angles
                    fja = finger_joint_angles(lms_w)

                    # Grasp
                    g_type, g_aperture, g_contact = classify_grasp(lms_w)
                    grasp_data[label] = (g_type, g_aperture, g_contact)
                    joint_data[label] = fja

                    # Smoothed dynamics on world-frame EE position
                    if label not in dynamics:
                        dynamics[label] = SmoothedDynamics(window=dyn_window)
                    ee_world = lms_world[0]
                    dynamics[label].update(ee_world, tc)
                    dyn = dynamics[label]

                    # CSV row
                    csv_row = build_csv_row(
                        fidx=fidx, label=label, tc=tc,
                        macro_task=macro_task, micro_step=micro_step,
                        step_idx=step_idx, total_steps=total_steps,
                        step_method=timeline.method,
                        op_height=op_height, robot_height=robot_height,
                        environment=environment, scene=scene,
                        lms_w=lms_w, lms_norm=lms_norm, lms_world=lms_world,
                        roll_cam=roll_c, pitch_cam=pitch_c, yaw_cam=yaw_c,
                        roll_world=roll_w, pitch_world=pitch_w, yaw_world=yaw_w,
                        ee_xyz=ee_xyz,
                        fja=fja,
                        g_type=g_type, g_aperture=g_aperture, g_contact=g_contact,
                        dyn=dyn,
                        imu_q=imu_q,
                        imu_roll=i_roll, imu_pitch=i_pitch, imu_yaw=i_yaw,
                        cam_accel_world=cam_accel_world,
                        cam_gyro=cam_gyro,
                    )
                    csv_w.write(csv_row)

                    # JSONL hand record
                    hand_json = build_hand_json(
                        label=label,
                        lms_w=lms_w, lms_norm=lms_norm, lms_world=lms_world,
                        roll_cam=roll_c, pitch_cam=pitch_c, yaw_cam=yaw_c,
                        roll_world=roll_w, pitch_world=pitch_w, yaw_world=yaw_w,
                        ee_xyz=ee_xyz,
                        fja=fja,
                        g_type=g_type, g_aperture=g_aperture, g_contact=g_contact,
                        dyn=dyn,
                    )
                    frame_hands_json.append(hand_json)

            # Pose
            pose_arr = None
            if p_res.pose_world_landmarks:
                pose_arr = pw(p_res.pose_world_landmarks[0])

            # JSONL frame record
            if frame_hands_json:
                frame_rec = build_jsonl_record(
                    video_name=input_path.name,
                    fidx=fidx, tc=tc,
                    macro_task=macro_task, micro_step=micro_step,
                    step_idx=step_idx, total_steps=total_steps,
                    step_method=timeline.method,
                    environment=environment, scene=scene,
                    op_height=op_height, robot_height=robot_height,
                    imu_state=imu_state,
                    hands_data=frame_hands_json,
                )
                jsonl_w.write(frame_rec)

            # Render annotated frame
            if writer:
                out_frame = build_frame(
                    frame=undistorted,
                    nlms=nlms, hness=hness,
                    cur_hands=cur_hands,
                    rpy_data=rpy_data,
                    grasp_data=grasp_data,
                    joint_data=joint_data,
                    pose_arr=pose_arr,
                    haw=haw,
                    out_w=out_w, out_h=out_h,
                    macro_task=macro_task,
                    micro_step=micro_step,
                    step_idx=step_idx,
                    total_steps=total_steps,
                    t0=ts_start, tc=tc,
                    nl_caption=nl_caption,
                    env=environment, scene=scene, oph=op_height,
                    imu_roll=i_roll, imu_pitch=i_pitch, imu_yaw=i_yaw,
                    imu_accel_history=list(imu_accel_hist),
                    step_method=timeline.method,
                )
                writer.write(out_frame)

            fidx += 1

        pbar.close()

    cap.release()
    if writer:
        writer.release()

    return {
        "video":  str(out_path) if write_video else None,
        "csv":    str(csv_path),
        "jsonl":  str(jsonl_path),
        "rows":   csv_w.rows_written,
        "frames": jsonl_w.frames_written,
        "imu_fused": imu_available,
        "step_method": timeline.method,
    }


# ── Batch processor ───────────────────────────────────────────────────────────

def process_batch(dataset_root: str | Path, output_dir: str | Path | None,
                  shared_kwargs: dict) -> list[dict]:
    root    = Path(dataset_root)
    out_dir = Path(output_dir) if output_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(root.glob("*.mp4")) + sorted(root.glob("*.MP4"))
    print(f"\nBatch mode: found {len(videos)} video(s) in {root}")

    results = []
    for vid in videos:
        stem = vid.stem
        imu  = vid.with_suffix(".imu")
        vts  = vid.with_suffix(".vts")
        stats = process_video(
            input_path = vid,
            imu_path   = imu  if imu.exists()  else None,
            vts_path   = vts  if vts.exists()  else None,
            csv_path   = out_dir / f"{stem}.csv",
            jsonl_path = out_dir / f"{stem}.jsonl",
            out_path   = out_dir / f"{stem}_annotated.mp4",
            **shared_kwargs,
        )
        results.append(stats)
        print(f"  Done: {stats}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_height(v: str) -> float:
    return float(v.lower().replace("cm","").strip())


def main():
    ap = argparse.ArgumentParser(
        description="VLA Egocentric Pipeline — fisheye + IMU fusion",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Mode
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input",  help="Single input video path (.mp4)")
    mode.add_argument("--batch",  help="Folder of videos for batch processing")

    # Companion files (single mode)
    ap.add_argument("--imu",  default=None, help="IMU data file (.imu)")
    ap.add_argument("--vts",  default=None, help="VTS sync file (.vts)")

    # Outputs
    ap.add_argument("--csv",        default="hand_6dof.csv")
    ap.add_argument("--jsonl",      default="vla_dataset.jsonl")
    ap.add_argument("--output",     default="vla_output.mp4")
    ap.add_argument("--output_dir", default=None,
                    help="Output directory (batch mode). Default = same as input folder.")
    ap.add_argument("--no_video",   action="store_true", help="Skip video rendering")

    # Video options
    ap.add_argument("--width",  type=int,   default=1600)
    ap.add_argument("--height", type=int,   default=720)
    ap.add_argument("--skip",   type=int,   default=1)
    ap.add_argument("--hands",  type=int,   default=2)
    ap.add_argument("--conf",   type=float, default=0.55)

    # Task labels
    ap.add_argument("--macro_task",  default="")
    ap.add_argument("--steps",       default="",
                    help="Semicolon-separated micro steps")
    ap.add_argument("--nl_caption",  default="")
    ap.add_argument("--ts_start",    type=float, default=0.0)
    ap.add_argument("--ts_end",      type=float, default=0.0)

    # Environment
    ap.add_argument("--environment",   default="Factory")
    ap.add_argument("--scene",         default="Workstation")
    ap.add_argument("--op_height",     type=_parse_height, default="170")
    ap.add_argument("--robot_height",  type=_parse_height, default="0")

    # Camera calibration overrides
    ap.add_argument("--cam_balance",   type=float, default=0.5,
                    help="Undistortion balance: 0=crop, 1=full FoV (default 0.5)")

    # IMU options
    ap.add_argument("--imu_beta",  type=float, default=0.033,
                    help="Madgwick beta parameter (default 0.033)")
    ap.add_argument("--no_mag",    action="store_true",
                    help="Disable magnetometer in IMU fusion")

    # Dynamics
    ap.add_argument("--dyn_window", type=int, default=9,
                    help="Savitzky-Golay window for velocity smoothing (odd, ≥5)")

    a = ap.parse_args()

    steps_list = [s.strip() for s in a.steps.split(";") if s.strip()] if a.steps else []

    shared = dict(
        out_w        = a.width,
        out_h        = a.height,
        skip         = a.skip,
        max_hands    = a.hands,
        conf         = a.conf,
        macro_task   = a.macro_task,
        steps        = steps_list,
        nl_caption   = a.nl_caption,
        ts_start     = a.ts_start,
        ts_end       = a.ts_end,
        environment  = a.environment,
        scene        = a.scene,
        op_height    = a.op_height,
        robot_height = a.robot_height,
        write_video  = not a.no_video,
        cam_balance  = a.cam_balance,
        imu_beta     = a.imu_beta,
        imu_mag      = not a.no_mag,
        dyn_window   = a.dyn_window,
    )

    if a.batch:
        results = process_batch(a.batch, a.output_dir, shared)
        total_rows   = sum(r["rows"]   for r in results)
        total_frames = sum(r["frames"] for r in results)
        print(f"\nBatch complete: {len(results)} video(s), "
              f"{total_rows:,} CSV rows, {total_frames:,} JSONL frames")
    else:
        # Auto-discover companion files if not specified
        input_p = Path(a.input)
        imu_p   = a.imu  or (str(input_p.with_suffix(".imu"))
                              if input_p.with_suffix(".imu").exists() else None)
        vts_p   = a.vts  or (str(input_p.with_suffix(".vts"))
                              if input_p.with_suffix(".vts").exists() else None)

        stats = process_video(
            input_path = a.input,
            imu_path   = imu_p,
            vts_path   = vts_p,
            csv_path   = a.csv,
            jsonl_path = a.jsonl,
            out_path   = a.output,
            **shared,
        )

        print(f"\nDone.")
        if stats["video"]:  print(f"  Video      : {stats['video']}")
        print(f"  CSV        : {stats['csv']}  ({stats['rows']:,} rows)")
        print(f"  JSONL      : {stats['jsonl']}  ({stats['frames']:,} frames)")
        print(f"  IMU fused  : {stats['imu_fused']}")
        print(f"  Step method: {stats['step_method']}")


if __name__ == "__main__":
    main()
