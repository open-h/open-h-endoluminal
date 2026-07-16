---
name: dataset-conversion
description: Help a contributor convert endoluminal robotics data (HDF5, Zarr, ROS bags, CSV plus frames) into the LeRobot v3.0 dataset format accepted by Open-H-Endoluminal. Use when asked to convert, format, or prepare data for submission.
---

# Dataset Conversion

Convert contributor source data into a LeRobot format v3.0 dataset ready for Open-H-Endoluminal submission. The v3.0 format requires the `lerobot` Python package v0.4.0 or later (the package version and the dataset format version are separate versioning schemes). Target layout: `meta/info.json` (codebase_version "v3.0"), `meta/stats.json`, `meta/tasks.parquet`, `meta/episodes/chunk-*/file-*.parquet`, data files that aggregate multiple episodes at `data/chunk-*/file-*.parquet`, and videos at `videos/<camera_key>/chunk-*/file-*.mp4`.

## Procedure

### 1. Inventory the source data

Establish before writing any code: directory layout and file formats; the streams present (video, kinematics, tracked pose, labels, language); sample rate of each stream; timestamp representation (units, epoch vs relative, clock source, dtype); physical units of state and action channels; and how episode boundaries are defined (one file per episode, an `episode_ends` array, bag boundaries, or something else).

### 2. Establish synchronization

Streams must share a common relative time base before conversion. Use `scripts/synchronization/` and the offset-then-sync pattern:

- `rosbag_parsing.py`: extract per-topic timestamps and values from ROS1 bags.
- `temp_cali.py`: estimate the constant time offset between two streams by sine-fitting a shared periodic motion.
- `post_sync.py`: apply per-stream offsets, then align messages across streams within a slop tolerance.

Final timestamps must be relative seconds from episode start (first timestamp at or near 0.0), strictly increasing, one distinct value per frame. Beware nanosecond sources (ROS `header.stamp`, many device SDKs): divide by 1e9 before use.

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
- `observation.state`: measured robot state (required). Flexible-robot convention: insertion depth, shaft rotation, tip bend angles (up-down, left-right) or tendon displacements, and tip pose. Catheters similar, with guidewire advance/retract and rotate.
- `observation.images.<view>`: each camera stream (examples: `observation.images.endoscope`, `observation.images.fluoro`).
- `observation.meta.<field>`: per-frame metadata (examples: `observation.meta.scope_type`, `observation.meta.em_pose` for auxiliary tracked pose).
- `observation.meta.camera_frame_delta_pose`: REQUIRED best effort for RGB endoscopy; the per-step relative camera pose expressed in the previous frame's optical coordinates (first frame is the identity). Derive it via `absolute_poses_to_camera_frame_deltas()` in `scripts/conversion/hdf5_to_lerobot.py`, applying the tip-to-camera calibration first; for monocular SLAM sources, state whether the scale is metric. Fluoroscopy-only conversions are exempt.
- `instruction.text`: timestep-level language.

Quality bars (suggested): >= 20 Hz, >= 480p, MP4 video encoding.

### 5. Set metadata and splits

- Per-episode task text stating task intent and target (example: "Navigate the colonoscope to the cecum"), never a generic label.
- `robot_type` and `fps` in the dataset creation call.
- `tolerance_s` recording the synchronization tolerance (typical 0.1 s).
- Optional but encouraged: first-class `recovery` and `failure` splits alongside train/val/test, recorded as episode-index ranges in `info.json` (see `custom_lerobot_split.py`).

### 6. Source already in LeRobot v2.1

Do not write a converter. Use the official conversion script that ships with LeRobot:

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id <id>
```

v2.1 artifacts (`episodes.jsonl`, `episodes_stats.jsonl`, `tasks.jsonl`, `data/chunk-*/episode_*.parquet`) must not remain in the output.

### 7. Complete the dataset README

Fill out `templates/dataset_template.md` and place it as `README.md` inside the dataset's `meta/` directory. The synchronization section (method, per-stream sample rates, measured skew) is required, as are task intent + target, device or platform, collection setting, signal tier, and licence and de-identification status.

### 8. Validate and iterate

```bash
python scripts/validation/validate_formatting.py <dataset_path> --verbose
```

Fix every ERROR and rerun until the report is clean; address WARNINGs where practical.

## Common pitfalls

- **Absolute epoch timestamps**: values near 1.7e9 are Unix epoch, not relative time. Subtract the episode start time.
- **float32 timestamps**: float32 keeps only ~7 significant digits, so epoch-scale values lose all sub-second precision and per-frame deltas collapse to zero. Store relative seconds, and use float64 if values can be large.
- **Nanoseconds**: timestamps 1e9 times too large make frame lookup select frame 0 forever. Convert to seconds first.
- **Non-ASCII feature keys**: feature names must be plain ASCII, dot-separated, exactly matching the conventions in step 4; smart quotes or accented characters break loaders.
- **Unlabeled navigation video**: raw, unlabeled video is not accepted. Every episode needs a stated task intent and target, and navigation data must show movement toward that target, not wandering.

Technical questions: Nigel Nelson, nigeln@nvidia.com.
