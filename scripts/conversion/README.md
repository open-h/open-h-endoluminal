# Data Conversion Scripts

This directory contains example scripts to convert endoluminal robotics datasets from common capture formats into the LeRobot dataset format v3.0, the standard format for Open-H-Endoluminal submissions.


The scripts are templates. Adapt the source key names, feature shapes, and state/action dimension names to your platform, but keep the Open-H-Endoluminal feature naming conventions: `action`, `observation.state`, camera streams as `observation.images.<view>` (e.g., `observation.images.endoscope`, `observation.images.fluoro`), per-frame metadata as `observation.meta.<field>`, and timestep-level language in `instruction.text`. Actions are positional setpoints (target positions and angles), not velocities. RGB endoscopy conversions should also populate `observation.meta.camera_frame_delta_pose` (a best-effort requirement; see `absolute_poses_to_camera_frame_deltas()` in `hdf5_to_lerobot.py` and the README section "Camera-Frame Kinematics for RGB Endoscopy").

**Timestamps (two timelines):** LeRobot's canonical `timestamp` column is always `frame_index / fps`. Preserve raw hardware clocks losslessly as `observation.meta.host_stamp_ns` (int64, Unix-epoch nanoseconds); `hdf5_to_lerobot.py` and `zarr_to_lerobot.py` show the pattern.

**Camera intrinsics:** static, so written once as `meta/calibration/camera_intrinsics.json` (keyed by camera feature name) rather than as a per-frame feature. Strongly encouraged for RGB endoscopy; `write_camera_intrinsics()` in `hdf5_to_lerobot.py` shows the pattern.

## Available Conversion Scripts

### `hdf5_to_lerobot.py`
Converts datasets from HDF5 format to LeRobot format. Designed for datasets where each HDF5 file represents a single episode with the structure:
- `/data/demo_0/action`: Positional actuation setpoints at each step (e.g., target insertion position, rotation angle, and tip-bend angles)
- `/data/demo_0/observations/rgb`: Endoscope RGB image observations
- `/data/demo_0/scope_state`: Flexible-endoscope state at each step (insertion depth, shaft rotation, tip bend up-down, tip bend left-right)
- `/data/demo_0/camera_pose`: Absolute camera (scope-tip) pose per step, `[x, y, z, qx, qy, qz, qw]` in any fixed world frame with the tip-to-camera calibration applied; the script converts it into the `observation.meta.camera_frame_delta_pose` feature via `absolute_poses_to_camera_frame_deltas()`
- `/data/demo_0/timestep`: Timestamps for each data point

### `zarr_to_lerobot.py`
Converts datasets from Zarr format to LeRobot format. Handles a single Zarr store containing multiple episodes concatenated into flat arrays (`action`, `observations/rgb`, `scope_state`, `timestep`), with episode boundaries defined by an `episode_ends` array.

### `custom_lerobot_split.py`
Demonstrates how to create custom dataset splits, including the optional first-class `recovery` and `failure` splits recorded as episode-index ranges in `meta/info.json`. It expects a root directory with `main/`, `recovery/`, and `failure/` subdirectories of HDF5 episode files. Recovery and failure demonstrations are valuable for training robust policies in safety-critical endoluminal procedures.

All three scripts use a tyro CLI (`--data-dir` or `--data-path`, `--repo-id`, `--push-to-hub`) and warn if you leave the default sentinel `--repo-id your-username/your-dataset-name` in place.

---

## Performance Optimization

### Video Encoding Parameters

LeRobot dataset creation supports several parameters that can significantly improve conversion performance for large datasets:

#### `image_writer_processes` and `image_writer_threads`
These parameters control parallel video encoding:
- **`image_writer_processes`**: Number of parallel processes for video encoding
- **`image_writer_threads`**: Number of threads per encoding process

**Performance Impact:**
- Default (no parallelization): ~947 seconds for a small dataset
- Optimized (15 threads, 10 processes): ~316 seconds (**3x faster**)

**Recommended Values:**
- `image_writer_processes=10-16` (adjust based on CPU cores)
- `image_writer_threads=15-20` (balance between throughput and memory usage)

#### `tolerance_s`
Time tolerance for data synchronization between different sensors (default: 0.1 seconds). Adjust based on your system's timing precision requirements, and record the value you used in your dataset README.

#### `batch_encoding_size` (Advanced)
Controls how many episodes are batched together before video encoding:
- **Benefits**: Further performance improvement (~8% faster)
- **Caveat**: Episodes in incomplete batches remain as individual images rather than MP4 videos
- **Recommendation**: Use only for large datasets where the batch size divides evenly into the total episode count

### Optimal Configuration Example

```python
dataset = LeRobotDataset.create(
    repo_id=repo_id,
    use_videos=True,
    robot_type="flexible_endoscope",
    fps=30,
    features={...},
    # Performance optimization parameters
    image_writer_processes=16,
    image_writer_threads=20,
    tolerance_s=0.1,
    # batch_encoding_size=12,  # Use with caution, see notes above
)
```

---

## Converting existing LeRobot v2.1 datasets

If you already collected data in the LeRobot v2.1 dataset format, use the official conversion script that ships with LeRobot 0.6.0:

```bash
# Dataset hosted on the Hugging Face Hub:
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id your-username/your-dataset-name

# Local-only dataset: --root is the dataset directory ITSELF (the folder
# containing meta/, data/, videos/); conversion happens in place and the
# original is preserved as a sibling <name>_old directory:
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id your-username/your-dataset-name \
    --root /path/to/your/dataset --push-to-hub false
```

---

## Validation

Every converted dataset must pass the format validator before submission:

```bash
python scripts/validation/validate_formatting.py /path/to/your/converted/dataset
```

Also remember that each submitted dataset must include a completed copy of `templates/dataset_template.md` as `README.md` inside its `meta/` directory, documenting the synchronization method and sample rates.
