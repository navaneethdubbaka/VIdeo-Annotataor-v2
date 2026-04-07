# Running the VLA pipeline

The main entry point is [`pipeline.py`](pipeline.py). It undistorts fisheye video, runs MediaPipe hand/pose tracking, optionally fuses **TRIMU001** `.imu` data and **TRIVTS** `.vts` frame timestamps, writes a multi-panel annotated video, and exports **CSV** + **JSONL** datasets.

## Setup

```bash
cd /path/to/Video_annotator_v2
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

On first run, MediaPipe task models are downloaded into the working directory (see `pipeline.py`).

## Modes

You must pass **either** `--input` (one video) **or** `--batch` (a folder of videos), not both.

| Mode | Flag | Behavior |
|------|------|----------|
| Single video | `--input path/to/recording.mp4` | One MP4; outputs default next to CWD unless you set paths |
| Batch | `--batch path/to/dataset_folder/` | Every `*.mp4` / `*.MP4` in the folder; uses `--output_dir` (default: same folder) |

### Companion files (single-video mode)

If you omit `--imu` / `--vts`, the pipeline looks next to the video for:

- `recording.imu` (same basename as the MP4)
- `recording.vts`

Batch mode always pairs `clip.mp4` with `clip.imu` / `clip.vts` when those files exist.

## Quick examples

**Single clip with defaults** (auto-pick `recording.imu` / `recording.vts` if present):

```bash
python pipeline.py --input "Video 1/recording1.mp4"
```

**Explicit sidecar paths:**

```bash
python pipeline.py --input recording.mp4 --imu recording.imu --vts recording.vts
```

**Batch folder** (writes `stem.csv`, `stem.jsonl`, `stem_annotated.mp4` per video):

```bash
python pipeline.py --batch ./dataset_root/ --output_dir ./outputs/
```

**No rendered video** (CSV + JSONL only):

```bash
python pipeline.py --input recording.mp4 --no_video
```

**Task labels and steps** (steps are semicolon-separated):

```bash
python pipeline.py --input recording.mp4 ^
  --macro_task "Disassemble a car wheel" ^
  --steps "Attach socket;Loosen nuts;Remove wheel" ^
  --nl_caption "Operator removes the wheel with an impact wrench."
```

## CLI flags

### Mode (required, mutually exclusive)

| Flag | Description |
|------|-------------|
| `--input` | Path to one input `.mp4` |
| `--batch` | Path to a folder containing videos |

### Companion files (single mode)

| Flag | Default | Description |
|------|---------|-------------|
| `--imu` | Auto: `<video_stem>.imu` if it exists | Binary/text IMU sidecar |
| `--vts` | Auto: `<video_stem>.vts` if it exists | Frame ↔ timestamp sync file |

### Outputs

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | `hand_6dof.csv` | Per-hand CSV path (single mode) |
| `--jsonl` | `vla_dataset.jsonl` | JSONL path (single mode) |
| `--output` | `vla_output.mp4` | Annotated video path (single mode) |
| `--output_dir` | Batch: input folder | Batch output directory |
| `--no_video` | off | Skip writing the composited MP4 |

### Video / detection

| Flag | Default | Description |
|------|---------|-------------|
| `--width` | `1600` | Composited output width (px) |
| `--height` | `720` | Composited output height (px) |
| `--skip` | `1` | Process every Nth frame (`1` = all processed frames) |
| `--hands` | `2` | Max hands for MediaPipe |
| `--conf` | `0.55` | Min hand detection confidence |

### Task / timeline

| Flag | Default | Description |
|------|---------|-------------|
| `--macro_task` | `""` | High-level task string |
| `--steps` | `""` | Micro steps, **semicolon-separated** |
| `--nl_caption` | `""` | Free-text caption |
| `--ts_start` | `0.0` | Step timeline start (seconds) |
| `--ts_end` | `0.0` | Step timeline end (`0` means use full video duration) |

### Environment / subject

| Flag | Default | Description |
|------|---------|-------------|
| `--environment` | `Factory` | Environment label |
| `--scene` | `Workstation` | Scene label |
| `--op_height` | `170` | Operator height (cm; suffix `cm` optional) |
| `--robot_height` | `0` | Robot height (cm) |

### Camera

| Flag | Default | Description |
|------|---------|-------------|
| `--cam_balance` | `0.5` | Fisheye undistortion balance: `0` = crop, `1` = full FoV |

### IMU fusion

| Flag | Default | Description |
|------|---------|-------------|
| `--imu_beta` | `0.033` | Madgwick β (used when fusion runs on stream) |
| `--no_mag` | off | Disable magnetometer in Madgwick updates |

### Other

| Flag | Default | Description |
|------|---------|-------------|
| `--dyn_window` | `9` | Savitzky–Golay window for dynamics (odd, ≥ 5) |
| `--limit` | `0` | Max frames to process (`0` = no limit) |

## Notes

- **IMU / VTS**: TRIMU001 **v2** (64-byte header, 76-byte samples) and TRIVTS **frame + `timestamp_ns`** records are supported; when both are present, timestamps are aligned using the IMU stream’s `time_origin_ns`.
- **Help**: `python pipeline.py -h` prints the same options with short descriptions.
