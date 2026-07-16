#!/usr/bin/env python
"""
An example script showing how to build a LeRobot dataset (format v3.0) with
custom splits, including the optional first-class "recovery" and "failure"
splits recorded as episode-index ranges in `meta/info.json`. This is an
Open-H best practice carried over from Open-H-Embodiment: recovery and
failure demonstrations are valuable for training robust policies in
safety-critical endoluminal procedures.

Version assumption:
-------------------
Requires the `lerobot` Python package v0.4.0 or later (`pip install "lerobot>=0.4.0"`).
Note that the package version and the dataset format version are separate
versioning schemes: lerobot >= 0.4.0 writes datasets in format v3.0.

Expected Directory Layout:
--------------------------
The script expects a root data directory with one subdirectory per episode
group, each containing HDF5 episode files with the structure documented in
`hdf5_to_lerobot.py` (adapt the key names to your own capture format):

<data-dir>/
    ├── main/        Successful navigation episodes (split into train/val/test).
    ├── recovery/    Recovery demonstrations (e.g., regaining a lost view).
    └── failure/     Failure examples.

Usage:
------
    python custom_lerobot_split.py --data-dir /path/to/your/data --repo-id your-username/your-dataset-name

To also push to the Hub:
    python custom_lerobot_split.py --data-dir /path/to/your/data --repo-id your-username/your-dataset-name --push-to-hub
"""

from pathlib import Path

import h5py
import tqdm
import tyro

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info


def add_episodes_from_dir(dataset, data_dir: Path, task_description: str) -> int:
    """Helper: read episodes from an HDF5 directory and add them to the dataset.

    Returns the number of episodes added.
    """
    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    print(f"Found {len(hdf5_files)} episodes in {data_dir}")

    for hdf5_path in tqdm.tqdm(hdf5_files, desc=f"Loading {data_dir.name}"):
        with h5py.File(hdf5_path, "r") as f:
            root_name = "data/demo_0"
            num_steps = len(f[f"{root_name}/action"])

            for step in range(num_steps):
                frame_data = {
                    "observation.images.endoscope": f[f"{root_name}/observations/rgb"][step],
                    "observation.state": f[f"{root_name}/scope_state"][step],
                    "action": f[f"{root_name}/action"][step],
                }
                timestamp = f[f"{root_name}/timestep"][step]
                dataset.add_frame(frame_data, task=task_description, timestamp=timestamp)

        dataset.save_episode()  # finalize this episode

    return len(hdf5_files)


def main(
    data_dir: Path = Path("path/to/your/data"),
    repo_id: str = "your-username/your-dataset-name",
    *,
    push_to_hub: bool = False,
):
    """
    Build a LeRobotDataset with train/val/test/recovery/failure splits.

    Args:
        data_dir: Root directory containing `main/`, `recovery/`, and `failure/`
            subdirectories of HDF5 episode files.
        repo_id: The desired Hugging Face Hub repository ID.
        push_to_hub: If True, uploads the dataset to the Hub after conversion.
    """
    if not data_dir.is_dir():
        print(f"Error: The provided data directory does not exist: {data_dir}")
        return

    if repo_id == "your-username/your-dataset-name":
        print("Warning: Using the default repo_id. Please specify your own with --repo-id.")

    # -----------------------
    # 1. Create fresh dataset
    # -----------------------
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
        },
        image_writer_processes=16,
        image_writer_threads=20,
        tolerance_s=0.1,
    )

    # -------------------------------
    # 2. Load main training episodes
    # -------------------------------
    num_main = add_episodes_from_dir(
        dataset,
        data_dir / "main",
        task_description="Navigate the colonoscope from the rectum to the cecum",
    )

    # -------------------------------
    # 3. Load recovery episodes
    # -------------------------------
    num_recovery = add_episodes_from_dir(
        dataset,
        data_dir / "recovery",
        task_description="Recover the endoscopic view and resume navigation to the cecum",
    )

    # -------------------------------
    # 4. Load failure episodes
    # -------------------------------
    num_failure = add_episodes_from_dir(
        dataset,
        data_dir / "failure",
        task_description="Navigate the colonoscope from the rectum to the cecum (failed attempt)",
    )

    # --------------------------------------
    # 5. Write custom splits into info.json
    # --------------------------------------
    # The main episodes are divided 70/15/15 into train/val/test. The recovery
    # and failure episodes follow as their own first-class splits, recorded as
    # episode-index ranges.
    train_end = int(0.70 * num_main)
    val_end = int(0.85 * num_main)
    dataset.meta.info["splits"] = {
        "train": f"0:{train_end}",
        "val": f"{train_end}:{val_end}",
        "test": f"{val_end}:{num_main}",
        "recovery": f"{num_main}:{num_main + num_recovery}",
        "failure": f"{num_main + num_recovery}:{num_main + num_recovery + num_failure}",
    }
    write_info(dataset.meta.info, dataset.root)

    print("Custom split configuration saved!")

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()
        print("Push complete.")


if __name__ == "__main__":
    tyro.cli(main)
