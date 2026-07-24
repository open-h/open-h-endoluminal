---
name: dataset-conversion
description: Help a contributor convert endoluminal robotics data (HDF5, Zarr, ROS bags, CSV plus frames) into the LeRobot v3.0 dataset format accepted by Open-H-Endoluminal. Use when asked to convert, format, or prepare data for submission.
---

# Dataset Conversion

Convert contributor source data into a LeRobot format v3.0 dataset ready for Open-H-Endoluminal submission. Use the pinned package `lerobot[dataset]==0.6.0` (Python >= 3.12; the `[dataset]` extra is required, and the package version and the dataset format version are separate versioning schemes). Target layout: `meta/info.json` (codebase_version "v3.0"), `meta/stats.json`, `meta/tasks.parquet`, `meta/episodes/chunk-*/file-*.parquet`, data files that aggregate multiple episodes at `data/chunk-*/file-*.parquet`, and videos at `videos/<camera_key>/chunk-*/file-*.mp4`.

## Procedure

### 1. Inventory the source data

Establish before writing any code: directory layout and file formats; the streams present (video, kinematics, tracked pose, labels, language); sample rate of each stream; timestamp representation (units, epoch vs relative, clock source, dtype); physical units of state and action channels; and how episode boundaries are defined (one file per episode, an `episode_ends` array, bag boundaries, or something else).

### 2. Establish synchronization

Streams must share a common relative time base before conversion. Use `scripts/synchronization/` and the offset-then-sync pattern:

- `rosbag_parsing.py`: extract per-topic timestamps and values from ROS1 bags.
- `temp_cali.py`: estimate the constant time offset between two streams by sine-fitting a shared periodic motion.
- `post_sync.py`: apply per-stream offsets, then align messages across streams within a slop tolerance.

Two timelines apply. (1) The canonical LeRobot `timestamp` column is the FRAME timeline: the library always writes `frame_index / fps` and does not accept explicit per-frame values, so streams must be captured or resampled at a fixed rate before conversion — that is what this synchronization step must deliver. (2) Ground-truth capture clocks are preserved losslessly as the pass-through feature `observation.meta.host_stamp_ns` (int64, Unix-epoch nanoseconds, one per frame; see the converters for the pattern). Beware unit mixups in sources (ROS `header.stamp` and many device SDKs are nanoseconds; others are seconds): convert to int64 nanoseconds for the meta feature.

### 3. Choose or adapt a converter

From `scripts/conversion/`, by source layout:

- `hdf5_to_lerobot.py`: a directory of HDF5 files, one episode per file.
- `zarr_to_lerobot.py`: a single Zarr store with flat concatenated arrays and an `episode_ends` boundary array.
- `custom_lerobot_split.py`: reference for writing custom splits (recovery, failure).
- ROS bags: parse and synchronize with `scripts/synchronization/`, then feed frames through the HDF5-style episode loop.
- CSV plus image frames: load CSV columns as state/action arrays, frames as the camera stream, then reuse the same episode loop.

Adapt the feature mapping and dataset creation parameters; see `scripts/conversion/README.md` for `image_writer_processes` / `image_writer_threads` performance tuning.

### 4. Map streams to the feature-naming conventions

- `action`: commanded values (required), as positional setpoints (target positions and angles), not velocities.
- `observation.state`: measured robot state (required) — the PRIMARY proprioception describing the robot/endoscope. Flexible-robot convention: insertion depth, shaft rotation, tip bend angles (up-down, left-right) or tendon displacements. Catheters similar, with guidewire advance/retract and rotate. Tip pose belongs here only when it is the primary proprioception (S2-only platforms with no native kinematics); when native kinematics fill `observation.state`, a tracked pose is auxiliary and goes to `observation.meta.em_pose` — do not concatenate it into `observation.state`.
- `observation.images.<view>`: each camera stream (examples: `observation.images.endoscope`, `observation.images.fluoro`).
- `observation.meta.<field>`: per-frame metadata (examples: `observation.meta.scope_type`, `observation.meta.em_pose` for auxiliary tracked pose).
- `observation.meta.camera_frame_delta_pose`: REQUIRED best effort for RGB endoscopy; the per-step relative camera pose expressed in the previous frame's optical coordinates (first frame is the identity). Derive it via `absolute_poses_to_camera_frame_deltas()` in `scripts/conversion/hdf5_to_lerobot.py`, applying the tip-to-camera calibration first; for monocular SLAM sources, state whether the scale is metric. Fluoroscopy-only conversions are exempt.
- Camera intrinsics: strongly encouraged for RGB endoscopy (required for the depth, 3D reconstruction, and SLAM uses the initiative targets). Intrinsics are static, so store them once as `meta/calibration/camera_intrinsics.json` keyed by camera feature name (OpenCV pinhole `fx`, `fy`, `cx`, `cy` in pixels, resolution, and distortion model plus coefficients), NOT as a per-frame feature. See `CAMERA_INTRINSICS` and `write_camera_intrinsics()` in `scripts/conversion/hdf5_to_lerobot.py`. If intrinsics genuinely vary per frame (an optical or digital zoom), store them as a per-frame `observation.meta.camera_intrinsics` feature instead.
- `observation.meta.host_stamp_ns`: ground-truth capture clock, int64 Unix-epoch nanoseconds, one per frame (the canonical `timestamp` column is always `frame_index / fps` — see step 2).
- `instruction.text`: timestep-level language.

Quality bars (suggested): >= 20 Hz, >= 480p, MP4 video encoding.

### 5. Set metadata and splits

- Per-episode task text stating task intent and target (example: "Navigate the colonoscope to the cecum"), never a generic label.
- `robot_type` and `fps` in the dataset creation call.
- `tolerance_s` recording the synchronization tolerance (typical 0.1 s).
- Optional but encouraged: first-class `recovery` and `failure` splits alongside train/val/test, recorded as episode-index ranges in `info.json` (see `custom_lerobot_split.py`).

### 6. Source already in LeRobot v2.1

Do not write a converter. Use the official conversion script that ships with LeRobot 0.6.0 (for local-only datasets add `--root <dataset_dir> --push-to-hub false`, where `--root` is the dataset directory ITSELF — the folder containing `meta/`, `data/`, `videos/`; conversion is in place, original preserved as a sibling `<name>_old`):

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id <id>
```

Before converting, drop or declare any parquet columns not listed in `meta/info.json` features (undeclared columns make the converted dataset unloadable), and note the converter does not carry over `meta/README.md` or `meta/calibration/` — add those after conversion.

v2.1 artifacts (`episodes.jsonl`, `episodes_stats.jsonl`, `tasks.jsonl`, `data/chunk-*/episode_*.parquet`) must not remain in the output.

### 7. Complete the dataset README

Fill out `templates/dataset_template.md` and place it as `README.md` inside the dataset's `meta/` directory. The synchronization section (method, per-stream sample rates, measured skew) is required, as are task intent + target, device or platform, collection setting, signal tier, and licence and de-identification status.

### 8. Validate and iterate

```bash
python scripts/validation/validate_formatting.py <dataset_path> --verbose
```

Fix every ERROR and rerun until the report is clean; address WARNINGs where practical.

## Common pitfalls

- **Absolute epoch timestamps in the canonical column**: `timestamp` values near 1.7e9 are Unix epoch, not the relative frame timeline — a symptom of a hand-rolled writer. Datasets written through `LeRobotDataset.add_frame` get `frame_index / fps` automatically; epoch clocks belong in `observation.meta.host_stamp_ns`.
- **float32 timestamps**: float32 keeps only ~7 significant digits, so epoch-scale values lose all sub-second precision and per-frame deltas collapse to zero. This is why ground-truth clocks are stored as int64 nanoseconds in the meta feature, never in the float32 `timestamp` column.
- **Nanosecond/second unit mixups**: know the unit of every source clock before converting to the int64-ns meta feature; a canonical `timestamp` column 1e9 times too large makes video frame lookup select frame 0 forever.
- **Non-ASCII feature keys**: feature names must be plain ASCII, dot-separated, exactly matching the conventions in step 4; smart quotes or accented characters break loaders.
- **Unlabeled navigation video**: raw, unlabeled video is not accepted. Every episode needs a stated task intent and target, and navigation data must show movement toward that target, not wandering.

Technical questions: Nigel Nelson, nigeln@nvidia.com.
