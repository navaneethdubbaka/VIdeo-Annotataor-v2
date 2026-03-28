"""
export.py — CSV and JSONL export with full enriched schema
===========================================================
Manages writing annotated records to:
  • CSV  — flat tabular format, one row per hand per frame
  • JSONL — nested JSON, one record per frame (multi-hand array)

New columns vs old pipeline:
  imu_qw/qx/qy/qz, cam_ax/ay/az_world, cam_gx/gy/gz,
  ee_jx/jy/jz/jerk, lm_world_x0..z20 (world-frame landmarks),
  imu_roll/pitch/yaw, step_boundary_method
"""

from __future__ import annotations
import csv, json
from pathlib import Path


# ── CSV header ────────────────────────────────────────────────────────────────

def _csv_header() -> list[str]:
    base = [
        "frame", "hand", "ts_sec",
        "macro_task", "micro_step", "step_idx", "total_steps",
        "step_boundary_method",
        "op_height_cm", "robot_height_cm",
        "environment", "scene",
    ]
    # Raw world landmarks (camera frame, Y-up)  x0..z20
    lm_raw = (
        [f"x{i}" for i in range(21)] +
        [f"y{i}" for i in range(21)] +
        [f"z{i}" for i in range(21)]
    )
    # Height-normalised landmarks
    lm_norm = (
        [f"nx{i}" for i in range(21)] +
        [f"ny{i}" for i in range(21)] +
        [f"nz{i}" for i in range(21)]
    )
    # World-frame landmarks (camera + IMU fused)
    lm_world = (
        [f"wx{i}" for i in range(21)] +
        [f"wy{i}" for i in range(21)] +
        [f"wz{i}" for i in range(21)]
    )
    # 6-DoF (camera-relative RPY)
    rpy_cam = ["roll_cam", "pitch_cam", "yaw_cam"]
    # 6-DoF (world-frame RPY, IMU-fused)
    rpy_world = ["roll_world", "pitch_world", "yaw_world"]
    # End-effector
    ee = ["ee_x", "ee_y", "ee_z"]
    # Joint angles
    joints = [
        "thumb_mcp","thumb_ip",
        "idx_mcp","idx_pip","idx_dip",
        "mid_mcp","mid_pip","mid_dip",
        "ring_mcp","ring_pip","ring_dip",
        "pinky_mcp","pinky_pip","pinky_dip",
    ]
    # Grasp
    grasp = ["grasp_type","finger_aperture_m","contact_state"]
    # Dynamics (velocity + acceleration + jerk)
    dynamics = [
        "ee_vx","ee_vy","ee_vz","ee_speed",
        "ee_ax","ee_ay","ee_az","ee_accel",
        "ee_jx","ee_jy","ee_jz","ee_jerk",
    ]
    # IMU state
    imu = [
        "imu_qw","imu_qx","imu_qy","imu_qz",
        "imu_roll","imu_pitch","imu_yaw",
        "cam_ax_world","cam_ay_world","cam_az_world",
        "cam_gx","cam_gy","cam_gz",
    ]
    return (base + lm_raw + lm_norm + lm_world +
            rpy_cam + rpy_world + ee + joints + grasp + dynamics + imu)


CSV_HEADER = _csv_header()


# ── Writers ───────────────────────────────────────────────────────────────────

class CSVWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f   = open(self.path, "w", newline="")
        self._w   = csv.writer(self._f)
        self._w.writerow(CSV_HEADER)
        self.rows_written = 0

    def write(self, rec: dict):
        """Write one dict (keys = CSV_HEADER) as a row."""
        row = [rec.get(k, "") for k in CSV_HEADER]
        self._w.writerow(row)
        self.rows_written += 1

    def close(self):
        self._f.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class JSONLWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f   = open(self.path, "w")
        self.frames_written = 0

    def write(self, rec: dict):
        self._f.write(json.dumps(rec, allow_nan=False) + "\n")
        self.frames_written += 1

    def close(self):
        self._f.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


# ── Record builders ───────────────────────────────────────────────────────────

def build_csv_row(
    fidx: int, label: str, tc: float,
    macro_task: str, micro_step: str, step_idx: int, total_steps: int,
    step_method: str,
    op_height: float, robot_height: float, environment: str, scene: str,
    lms_w: "np.ndarray",        # (21,3) camera-frame world landmarks
    lms_norm: "np.ndarray",     # (21,3) height-normalised
    lms_world: "np.ndarray",    # (21,3) IMU-rotated world frame
    roll_cam: float, pitch_cam: float, yaw_cam: float,
    roll_world: float, pitch_world: float, yaw_world: float,
    ee_xyz: "np.ndarray",
    fja: dict,
    g_type: str, g_aperture: float, g_contact: str,
    dyn: "SmoothedDynamics",
    imu_q: "np.ndarray",        # [w,x,y,z]
    imu_roll: float, imu_pitch: float, imu_yaw: float,
    cam_accel_world: "np.ndarray",
    cam_gyro: "np.ndarray",
) -> dict:
    """Build a flat dict keyed to CSV_HEADER from all computed features."""
    import numpy as np

    def r6(v): return round(float(v), 6)
    def r3(v): return round(float(v), 3)

    row: dict = {
        "frame": fidx, "hand": label, "ts_sec": f"{tc:.3f}",
        "macro_task": macro_task, "micro_step": micro_step,
        "step_idx": step_idx, "total_steps": total_steps,
        "step_boundary_method": step_method,
        "op_height_cm": op_height, "robot_height_cm": robot_height,
        "environment": environment, "scene": scene,
        "roll_cam": r3(roll_cam), "pitch_cam": r3(pitch_cam), "yaw_cam": r3(yaw_cam),
        "roll_world": r3(roll_world), "pitch_world": r3(pitch_world), "yaw_world": r3(yaw_world),
        "ee_x": r6(ee_xyz[0]), "ee_y": r6(ee_xyz[1]), "ee_z": r6(ee_xyz[2]),
        "grasp_type": g_type,
        "finger_aperture_m": r6(g_aperture),
        "contact_state": g_contact,
        "imu_qw": r6(imu_q[0]), "imu_qx": r6(imu_q[1]),
        "imu_qy": r6(imu_q[2]), "imu_qz": r6(imu_q[3]),
        "imu_roll": r3(imu_roll), "imu_pitch": r3(imu_pitch), "imu_yaw": r3(imu_yaw),
        "cam_ax_world": r6(cam_accel_world[0]),
        "cam_ay_world": r6(cam_accel_world[1]),
        "cam_az_world": r6(cam_accel_world[2]),
        "cam_gx": r6(cam_gyro[0]), "cam_gy": r6(cam_gyro[1]), "cam_gz": r6(cam_gyro[2]),
    }

    # Landmarks
    for i in range(21):
        row[f"x{i}"]  = r6(lms_w[i,0]);    row[f"y{i}"]  = r6(lms_w[i,1]);    row[f"z{i}"]  = r6(lms_w[i,2])
        row[f"nx{i}"] = r6(lms_norm[i,0]);  row[f"ny{i}"] = r6(lms_norm[i,1]); row[f"nz{i}"] = r6(lms_norm[i,2])
        row[f"wx{i}"] = r6(lms_world[i,0]); row[f"wy{i}"] = r6(lms_world[i,1]);row[f"wz{i}"] = r6(lms_world[i,2])

    # Joint angles
    for k, v in fja.items():
        row[k] = r3(v)

    # Dynamics
    row.update(dyn.as_dict())

    return row


def build_jsonl_record(
    video_name: str,
    fidx: int, tc: float,
    macro_task: str, micro_step: str, step_idx: int, total_steps: int,
    step_method: str,
    environment: str, scene: str,
    op_height: float, robot_height: float,
    imu_state,          # IMUSample
    hands_data: list[dict],
) -> dict:
    """Build one JSONL frame record (hands is a list of per-hand dicts)."""
    q = imu_state.quaternion
    g = imu_state.accel_world
    roll, pitch, yaw = imu_state.euler_deg()

    return {
        "video":       video_name,
        "frame":       fidx,
        "ts_sec":      round(tc, 3),
        "macro_task":  macro_task,
        "micro_step":  micro_step,
        "step_idx":    step_idx,
        "total_steps": total_steps,
        "step_boundary_method": step_method,
        "environment": environment,
        "scene":       scene,
        "op_height_cm":    op_height,
        "robot_height_cm": robot_height,
        "imu": {
            "quaternion":              [round(float(v),6) for v in q],
            "roll_deg":                round(roll,  3),
            "pitch_deg":               round(pitch, 3),
            "yaw_deg":                 round(yaw,   3),
            "gravity_subtracted_accel":[round(float(v),6) for v in g],
            "gyro":                    [round(float(v),6) for v in
                                        [imu_state.gx, imu_state.gy, imu_state.gz]],
        },
        "hands": hands_data,
    }


def build_hand_json(
    label: str,
    lms_w: "np.ndarray",
    lms_norm: "np.ndarray",
    lms_world: "np.ndarray",
    roll_cam: float, pitch_cam: float, yaw_cam: float,
    roll_world: float, pitch_world: float, yaw_world: float,
    ee_xyz: "np.ndarray",
    fja: dict,
    g_type: str, g_aperture: float, g_contact: str,
    dyn: "SmoothedDynamics",
) -> dict:
    def r6(v): return round(float(v), 6)
    def r3(v): return round(float(v), 3)

    return {
        "hand":          label,
        "xyz_cam":       [[r6(v) for v in row] for row in lms_w.tolist()],
        "xyz_norm":      [[r6(v) for v in row] for row in lms_norm.tolist()],
        "xyz_world":     [[r6(v) for v in row] for row in lms_world.tolist()],
        "rpy_cam":       [r3(roll_cam),   r3(pitch_cam),   r3(yaw_cam)],
        "rpy_world":     [r3(roll_world), r3(pitch_world), r3(yaw_world)],
        "ee_xyz":        [r6(ee_xyz[0]), r6(ee_xyz[1]), r6(ee_xyz[2])],
        "joint_angles":  {k: r3(v) for k,v in fja.items()},
        "grasp_type":    g_type,
        "finger_aperture_m": r6(g_aperture),
        "contact_state": g_contact,
        **dyn.as_dict(),
    }
