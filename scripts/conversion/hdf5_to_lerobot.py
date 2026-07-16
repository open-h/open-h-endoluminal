#!/usr/bin/env python
"""
A script to convert endoluminal robotics data from HDF5 files into the LeRobot
dataset format v3.0 with an efficient MP4 video backend.

Version assumption:
-------------------
Requires the `lerobot` Python package v0.4.0 or later (`pip install "lerobot>=0.4.0"`).
Note that the package version and the dataset format version are separate
versioning schemes: lerobot >= 0.4.0 writes datasets in format v3.0.

This script processes a directory of HDF5 files, where each file represents a
single episode. It extracts observations, actions, and state information, and
packages them into a LeRobotDataset with visual data stored as compressed MP4
videos, then optionally pushes the result to the Hugging Face Hub.

Expected HDF5 File Structure:
------------------------------
The script assumes a directory with zero-indexed HDF5 files (e.g., `data_0.hdf5`).
Each file should contain the following structure (adapt the key names below to
match your own capture format):

/data/demo_0/
    ├── action                (Dataset): Positional actuation setpoints at each step
    │                         (target positions and angles, not velocities).
    ├── observations/
    │   └── rgb               (Dataset): Endoscope RGB image observations.
    ├── scope_state           (Dataset): Flexible-endoscope state at each step.
    ├── camera_pose           (Dataset): Absolute camera (scope-tip) pose per step,
    │                         [x, y, z, qx, qy, qz, qw] in any fixed world frame,
    │                         with the tip-to-camera calibration already applied.
    └── timestep              (Dataset): Timestamps for each data point.

The camera_pose stream feeds the 'observation.meta.camera_frame_delta_pose'
feature (REQUIRED best effort for RGB endoscopy): see
absolute_poses_to_camera_frame_deltas() below and the README section
"Camera-Frame Kinematics for RGB Endoscopy". Additional dependencies for
that computation: numpy and scipy.

Usage:
------
    python hdf5_to_lerobot.py --data-dir /path/to/your/hdf5/files --repo-id your-username/your-dataset-name

To also push to the Hub:
    python hdf5_to_lerobot.py --data-dir /path/to/your/hdf5/files --repo-id your-username/your-dataset-name --push-to-hub
"""

import shutil
from pathlib import Path

import h5py
import numpy as np
import tqdm
import tyro
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.constants import HF_LEROBOT_HOME


def absolute_poses_to_camera_frame_deltas(poses):
    """Convert absolute camera poses to per-step camera-frame delta poses.

    This is the reference implementation of the Open-H-Endoluminal
    'observation.meta.camera_frame_delta_pose' feature: the pose of camera
    frame i expressed in camera frame i-1 (OpenCV optical convention:
    +x right, +y down, +z along the optical axis). Because the chip-on-tip
    camera is the end effector in endoluminal robotics, this is the
    equivalent of the camera-frame end-effector pose used by rigid-arm
    datasets such as Open-X Embodiment.

    Args:
        poses: (N, 7) array of absolute camera poses [x, y, z, qx, qy, qz, qw]
            in any fixed world frame. Apply your tip-to-camera (hand-eye) or
            sensor-to-camera calibration first, so the poses describe the
            camera itself, not the tracking sensor or the tip body frame.

    Returns:
        (N, 7) float32 array [dx, dy, dz, dqx, dqy, dqz, dqw], with row 0 set
        to the identity transform [0, 0, 0, 0, 0, 0, 1].
    """
    poses = np.asarray(poses, dtype=np.float64)
    deltas = np.zeros((len(poses), 7), dtype=np.float32)
    deltas[:, 6] = 1.0  # identity quaternion for the first frame
    if len(poses) < 2:
        return deltas

    rotations = Rotation.from_quat(poses[:, 3:7])
    for i in range(1, len(poses)):
        previous_inverse = rotations[i - 1].inv()
        deltas[i, :3] = previous_inverse.apply(poses[i, :3] - poses[i - 1, :3])
        deltas[i, 3:7] = (previous_inverse * rotations[i]).as_quat()
    return deltas


def convert_data_to_lerobot(data_dir: Path, repo_id: str, *, push_to_hub: bool = False):
    """
    Converts a directory of HDF5 files to a LeRobotDataset with a video backend.

    Args:
        data_dir: The path to the directory containing the HDF5 files.
        repo_id: The repository ID for the dataset on the Hugging Face Hub.
        push_to_hub: Whether to push the dataset to the Hub after conversion.
    """
    final_output_path = Path(HF_LEROBOT_HOME) / repo_id
    if final_output_path.exists():
        print(f"Removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        use_videos=True,
        robot_type="flexible_endoscope",
        fps=30,
        features={
            # Adjust the shape to your capture resolution.
            # Suggested quality bar: >= 480p at >= 20 Hz, MP4 encoding.
            "observation.images.endoscope": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            # If your platform has a second camera stream (for example an
            # external bench camera or a fluoroscopy view), add it here as
            # another "observation.images.<view>" entry, e.g.
            # "observation.images.fluoro".
            "observation.state": {
                "dtype": "float32",
                "shape": (4,),
                "names": [
                    "insertion_depth",
                    "shaft_rotation",
                    "tip_bend_up_down",
                    "tip_bend_left_right",
                ],
            },
            # Positional setpoints (target positions and angles), not velocities.
            "action": {
                "dtype": "float32",
                "shape": (4,),
                "names": [
                    "insertion_setpoint",
                    "rotation_setpoint",
                    "bend_up_down_setpoint",
                    "bend_left_right_setpoint",
                ],
            },
            # REQUIRED best effort for RGB endoscopy: camera-frame motion, the
            # endoluminal equivalent of the camera-frame end-effector pose used
            # by rigid-arm datasets. See the README section "Camera-Frame
            # Kinematics for RGB Endoscopy". Drop this feature only if it is
            # genuinely infeasible for your platform, and justify that in your
            # dataset's meta/README.md.
            "observation.meta.camera_frame_delta_pose": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["dx_m", "dy_m", "dz_m", "dqx", "dqy", "dqz", "dqw"],
            },
        },
        image_writer_processes=16,
        image_writer_threads=20,
        tolerance_s=0.1,
    )

    hdf5_files = sorted(data_dir.glob("*.hdf5"))

    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}. Exiting.")
        return

    print(f"Found {len(hdf5_files)} episodes to convert.")

    task_description = "Navigate the colonoscope from the rectum to the cecum"

    for hdf5_path in tqdm.tqdm(hdf5_files, desc="Converting Episodes"):
        try:
            with h5py.File(hdf5_path, "r") as f:
                root_name = "data/demo_0"
                if root_name not in f:
                    print(f"Warning: Skipping {hdf5_path} because '{root_name}' group was not found.")
                    continue

                num_steps = len(f[f"{root_name}/action"])

                # Convert this episode's absolute camera poses into per-step
                # camera-frame delta poses (first frame is the identity).
                camera_frame_deltas = absolute_poses_to_camera_frame_deltas(
                    f[f"{root_name}/camera_pose"][:]
                )

                # Add each frame from the episode to the internal buffer.
                for step in range(num_steps):
                    frame_data = {
                        "observation.images.endoscope": f[f"{root_name}/observations/rgb"][step],
                        "observation.state": f[f"{root_name}/scope_state"][step],
                        "action": f[f"{root_name}/action"][step],
                        "observation.meta.camera_frame_delta_pose": camera_frame_deltas[step],
                    }
                    timestamp = f[f"{root_name}/timestep"][step]
                    dataset.add_frame(frame_data, task=task_description, timestamp=timestamp)

            # After processing all frames for an HDF5 file, save the buffered
            # data as a completed episode. This will trigger the video encoding
            # for the endoscope frames collected.
            dataset.save_episode()

        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")
            # It's good practice to clear the buffer on error to prevent
            # a failed episode from contaminating the next one.
            dataset.clear_episode_buffer()

    print(f"Dataset conversion complete. Saved to {final_output_path}")

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()
        print("Push complete.")


def main(
    data_dir: Path = Path("path/to/your/data"),
    repo_id: str = "your-username/your-dataset-name",
    *,
    push_to_hub: bool = False,
):
    """
    Main entry point for the conversion script.

    Args:
        data_dir: The directory containing HDF5 episode files.
        repo_id: The desired Hugging Face Hub repository ID.
        push_to_hub: If True, uploads the dataset to the Hub.
    """
    if not data_dir.is_dir():
        print(f"Error: The provided data directory does not exist: {data_dir}")
        return

    if repo_id == "your-username/your-dataset-name":
        print("Warning: Using the default repo_id. Please specify your own with --repo-id.")

    convert_data_to_lerobot(data_dir, repo_id, push_to_hub=push_to_hub)


if __name__ == "__main__":
    tyro.cli(main)
