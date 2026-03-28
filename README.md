# VLA Egocentric Data Collection Pipeline

Egocentric RGB fisheye video → 9-DoF IMU fusion → world-frame kinematic annotation → VLA-ready dataset export.

## Files

| File | Purpose |
|------|---------|
| `pipeline.py`    | Main entry point — CLI + orchestration |
| `camera.py`      | Fisheye undistortion (Kannala-Brandt) |
| `imu.py`         | IMU parsing + Madgwick AHRS fusion |
| `vts_parser.py`  | Auto-detecting .vts binary parser |
| `kinematics.py`  | 6-DoF math, grasp, joint angles, dynamics |
| `segmentation.py`| IMU-guided step timeline |
| `compositor.py`  | 4-panel annotated video renderer |
| `export.py`      | CSV + JSONL writers |

## Install

```bash
pip install -r requirements.txt
```

## Usage — single video

```bash
python pipeline.py \
    --input  recording.mp4 \
    --imu    recording.imu \
    --vts    recording.vts \
    --macro_task "Disassemble a car wheel" \
    --steps  "Attach socket;Loosen nuts;Remove wheel" \
    --nl_caption "Operator disassembles a car wheel with an impact wrench." \
    --environment "Car Workshop" \
    --scene "Car service" \
    --op_height 162 \
    --robot_height 120 \
    --skip 2 \
    --output annotated.mp4 \
    --csv    output.csv \
    --jsonl  output.jsonl
```

Companion files (`.imu`, `.vts`) are auto-discovered if they share the same stem as the video.

## Usage — batch folder

```bash
python pipeline.py \
    --batch /path/to/recordings/ \
    --output_dir /path/to/outputs/ \
    --macro_task "Assembly task" \
    --steps "Step 1;Step 2;Step 3" \
    --op_height 170
```

Expected folder structure:
```
recordings/
    clip001.mp4
    clip001.imu
    clip001.vts     ← optional
    clip002.mp4
    clip002.imu
    ...
```

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--skip N`        | 1       | Process every Nth frame |
| `--cam_balance`   | 0.5     | Undistortion balance (0=crop, 1=full FoV) |
| `--imu_beta`      | 0.033   | Madgwick convergence speed |
| `--no_mag`        | off     | Disable magnetometer (yaw will drift) |
| `--dyn_window`    | 9       | Savitzky-Golay window for velocity (odd) |
| `--no_video`      | off     | Skip video rendering (CSV/JSONL only) |
| `--robot_height`  | 0       | Target robot height cm (0 = no normalisation) |

## CSV schema (key new columns vs old pipeline)

| Column group | Columns | Description |
|---|---|---|
| World landmarks | `wx0..wz20` | Hand landmarks in world frame (IMU-rotated) |
| World 6-DoF     | `roll_world`, `pitch_world`, `yaw_world` | Hand orientation in gravity-aligned world frame |
| IMU orientation | `imu_qw/qx/qy/qz`, `imu_roll/pitch/yaw` | Camera world orientation from Madgwick |
| IMU dynamics    | `cam_ax/ay/az_world`, `cam_gx/gy/gz` | Camera acceleration + angular velocity |
| Jerk            | `ee_jx/jy/jz`, `ee_jerk` | End-effector jerk (3rd derivative) |
| Step method     | `step_boundary_method` | `imu_detected` or `equal_split` |

## JSONL schema

One JSON object per frame:
```json
{
  "video": "clip001.mp4",
  "frame": 42,
  "ts_sec": 1.4,
  "macro_task": "...",
  "micro_step": "...",
  "step_idx": 1,
  "total_steps": 3,
  "step_boundary_method": "imu_detected",
  "imu": {
    "quaternion": [w, x, y, z],
    "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
    "gravity_subtracted_accel": [ax, ay, az],
    "gyro": [gx, gy, gz]
  },
  "hands": [
    {
      "hand": "Right",
      "xyz_cam":   [[...21 landmarks, camera frame...]],
      "xyz_norm":  [[...height-normalised...]],
      "xyz_world": [[...world frame, IMU-rotated...]],
      "rpy_cam":   [roll, pitch, yaw],
      "rpy_world": [roll, pitch, yaw],
      "ee_xyz":    [x, y, z],
      "joint_angles": {"thumb_mcp": 0.0, ...},
      "grasp_type": "power",
      "finger_aperture_m": 0.05,
      "contact_state": "partial",
      "ee_vx": 0.0, "ee_vy": 0.0, "ee_vz": 0.0, "ee_speed": 0.0,
      "ee_ax": 0.0, "ee_ay": 0.0, "ee_az": 0.0, "ee_accel": 0.0,
      "ee_jx": 0.0, "ee_jy": 0.0, "ee_jz": 0.0, "ee_jerk": 0.0
    }
  ]
}
```

## Improvements over old pipeline

| Old | New | Impact |
|-----|-----|--------|
| Raw fisheye into MediaPipe | Kannala-Brandt undistortion first | Correct landmark positions |
| Camera-relative RPY only | World-frame RPY via IMU fusion | Robot-transferable orientation |
| Equal-duration step splits | IMU motion-valley detection | Steps match actual task structure |
| Raw finite-diff velocity | Savitzky-Golay smoothed + jerk | Cleaner dynamics signal |
| No IMU data exported | Full 9-DoF per frame in CSV+JSONL | Richer policy learning signal |
| .vts ignored | Auto-detected + used for sync | Better timestamp alignment |
