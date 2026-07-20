#!/usr/bin/env python
"""
A script to convert endoluminal robotics data from a single Zarr store into the
LeRobot dataset format v3.0.

Version assumption:
-------------------
Requires the `lerobot` Python package pinned at 0.6.0 with the dataset extra
(`pip install "lerobot[dataset]==0.6.0"`; Python >= 3.12).

This script is designed to process a single Zarr store that contains an entire
dataset, with episode boundaries defined by an `episode_ends` array. It extracts
observations, actions, and state information for each episode and packages them
into a LeRobotDataset, which can then be optionally pushed to the Hugging Face Hub.

Expected Zarr Store Structure:
------------------------------
The script assumes a single Zarr store (e.g., `my_dataset.zarr`) with the following
internal hierarchy (adapt the key names to match your own capture format). All
top-level arrays are expected to be flat, containing data for all episodes
concatenated together. `N` is the total number of steps across all episodes,
and `E` is the total number of episodes.

/
├── action                (Array, shape: (N, 4)): The actuation commands for all steps.
├── observations/
│   └── rgb               (Array, shape: (N, 480, 640, 3)): Endoscope RGB images for all steps.
├── scope_state           (Array, shape: (N, 4)): The flexible-endoscope state for all steps.
├── timestep              (Array, shape: (N,)): The timestamp for each data point.
└── episode_ends          (Array, shape: (E,)): Indices marking the end of each episode.

Usage:
------
To run the script, you can use the following command, pointing to your Zarr store:

    python zarr_to_lerobot.py --data-path /path/to/your/dataset.zarr --repo-id your-username/your-dataset-name

To convert and then upload to the Hugging Face Hub:

    python zarr_to_lerobot.py --data-path /path/to/your/dataset.zarr --repo-id your-username/your-dataset-name --push-to-hub

Dependencies:
-------------
- lerobot[dataset] == 0.6.0
- tyro
- zarr
- tqdm
"""

import shutil
from pathlib import Path

import numpy as np
import tqdm
import tyro
import zarr
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def convert_data_to_lerobot(data_path: Path, repo_id: str, *, push_to_hub: bool = False):
    """
    Converts a single Zarr store with episode boundaries to a LeRobotDataset.

    Args:
        data_path: The path to the Zarr store file/directory.
        repo_id: The repository ID for the dataset on the Hugging Face Hub.
        push_to_hub: Whether to push the dataset to the Hub after conversion.
    """
    final_output_path = Path(HF_LEROBOT_HOME) / repo_id
    if final_output_path.exists():
        print(f"Removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    # Initialize a LeRobotDataset with the desired features.
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
            # If your platform has a second camera stream (for example a
            # fluoroscopy view), add it here as another
            # "observation.images.<view>" entry, e.g. "observation.images.fluoro".
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
            # If you also record absolute camera poses, convert them into the
            # "observation.meta.camera_frame_delta_pose" feature (REQUIRED best
            # effort for RGB endoscopy); see absolute_poses_to_camera_frame_deltas()
            # in hdf5_to_lerobot.py and the README section "Camera-Frame
            # Kinematics for RGB Endoscopy".
            # Ground-truth capture clock, preserved losslessly. The canonical
            # LeRobot `timestamp` column is always frame_index / fps, so the
            # raw hardware stamps live in this pass-through feature: int64
            # Unix-epoch nanoseconds, one per frame.
            "observation.meta.host_stamp_ns": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["host_stamp_ns"],
            },
        },
        image_writer_processes=16,
        image_writer_threads=20,
        tolerance_s=0.1,
    )

    print(f"Opening Zarr store at {data_path}")
    try:
        root_zarr = zarr.open(store=str(data_path), mode='r')
    except Exception as e:
        print(f"Error opening Zarr store: {e}")
        return

    if "episode_ends" not in root_zarr:
        print(f"Error: `episode_ends` array not found in {data_path}. Cannot determine episode boundaries.")
        return

    episode_ends = root_zarr["episode_ends"][:]
    num_episodes = len(episode_ends)
    print(f"Found {num_episodes} episodes to convert.")

    # A single, descriptive task for all episodes. Every Open-H-Endoluminal
    # submission must state the task intent and the navigation target.
    task_description = "Navigate the colonoscope from the rectum to the cecum"

    # Process each episode based on the episode_ends indices.
    start_idx = 0
    for episode_idx in tqdm.tqdm(range(num_episodes), desc="Converting Episodes"):
        end_idx = episode_ends[episode_idx]
        try:
            # Convert the source capture clock to int64 epoch nanoseconds for
            # observation.meta.host_stamp_ns (assumes `timestep` holds
            # seconds; adapt if your store uses another unit).
            episode_stamp_ns = np.round(
                np.asarray(root_zarr["timestep"][start_idx:end_idx], dtype=np.float64) * 1e9
            ).astype(np.int64)

            # Add each frame from the current episode slice to the dataset
            # buffer. lerobot >= 0.4 takes a single dict with the task string
            # INSIDE it; the canonical `timestamp` column is synthesized as
            # frame_index / fps.
            for step_idx in range(start_idx, end_idx):
                # lerobot validates feature dtypes strictly: cast numeric
                # sources to the declared float32 explicitly.
                frame_data = {
                    "observation.images.endoscope": root_zarr["observations/rgb"][step_idx],
                    "observation.state": np.asarray(
                        root_zarr["scope_state"][step_idx], dtype=np.float32),
                    "action": np.asarray(
                        root_zarr["action"][step_idx], dtype=np.float32),
                    "observation.meta.host_stamp_ns": episode_stamp_ns[step_idx - start_idx : step_idx - start_idx + 1],
                    "task": task_description,
                }
                dataset.add_frame(frame_data)

            # Save the buffered frames as a completed episode.
            dataset.save_episode()

        except Exception as e:
            print(f"Error processing episode {episode_idx}: {e}")
            dataset.clear_episode_buffer()

        finally:
            # Always advance to the next episode boundary, even if this
            # episode failed, so a failure never bleeds into the next slice.
            start_idx = end_idx

    # REQUIRED: close the parquet writers. Without this the footer metadata
    # is never written and the resulting dataset may not load at all.
    dataset.finalize()

    print(f"Dataset conversion complete. Saved to {final_output_path}")

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()
        print("Push complete.")


def main(
    data_path: Path = Path("path/to/your/dataset.zarr"),
    repo_id: str = "your-username/your-dataset-name",
    *,
    push_to_hub: bool = False,
):
    """
    Main entry point for the conversion script.

    Args:
        data_path: The path to the single Zarr store for the dataset.
        repo_id: The desired Hugging Face Hub repository ID (e.g., 'username/dataset-name').
        push_to_hub: If True, uploads the dataset to the Hub after conversion.
    """
    if not data_path.exists():
        print(f"Error: The provided Zarr store does not exist: {data_path}")
        print("Please provide a valid path to your Zarr store.")
        return

    if repo_id == "your-username/your-dataset-name":
        print("Warning: Using the default repo_id. Please specify your own with --repo-id.")

    convert_data_to_lerobot(data_path, repo_id, push_to_hub=push_to_hub)


if __name__ == "__main__":
    tyro.cli(main)
