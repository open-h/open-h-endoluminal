# AGENTS.md: Operating Guide for AI Agents

## Project Orientation

Open-H-Endoluminal is an open, multi-institution dataset initiative for endoscopic, endoluminal, and interventional robotics: flexible GI scopes, bronchoscopes, ureteroscopes, capsules, continuum and soft robots, and vascular catheters that navigate the body's natural lumens and vasculature. It assembles synchronized, multimodal recordings (video, robot state and action, language) to train and evaluate vision-language-action models, world and simulation models, and core perception capabilities (depth estimation, 3D reconstruction and SLAM in lumens, segmentation, workflow understanding). It grows out of Open-H-Embodiment, the first large-scale open dataset for healthcare robotics (the basis of the GR00T-H vision-language-action model and the Cosmos-H world-model line). This repo is the contributor-facing home: the contribution guide, the RFP, conversion, synchronization, and validation scripts, the dataset README template, and agent skills. Datasets standardize on the LeRobot dataset format v3.0 and are released under CC BY 4.0.

## Repo Map

| Path | Purpose |
| :--- | :--- |
| `README.md` | Contribution guide: scope, task hierarchy, format requirements, submission funnel; links to the RFP for signal tiers, collection settings, and required metadata. |
| `assets/open-h-endoluminal-rfp.pdf` | The canonical, finalized RFP. DO NOT MODIFY (see note below). |
| `templates/dataset_template.md` | Dataset README template; contributors complete a copy as `meta/README.md` inside their dataset. |
| `scripts/conversion/README.md` | Overview of the converters plus video-encoding performance tuning notes. |
| `scripts/conversion/hdf5_to_lerobot.py` | Converts a directory of per-episode HDF5 files to a LeRobot dataset. |
| `scripts/conversion/zarr_to_lerobot.py` | Converts a single Zarr store (episode boundaries via `episode_ends`) to a LeRobot dataset. |
| `scripts/conversion/custom_lerobot_split.py` | Example of recording custom splits, including recovery and failure splits. |
| `scripts/synchronization/rosbag_parsing.py` | Extracts time-series data from selected ROS1 bag topics. |
| `scripts/synchronization/temp_cali.py` | Estimates the temporal offset between two periodic signals via sine-wave fitting. |
| `scripts/synchronization/post_sync.py` | Applies per-topic time offsets and aligns multi-topic messages within a slop tolerance. |
| `scripts/validation/validate_formatting.py` | Local compliance validator; the final check before a dataset is submitted. |
| `AGENTS.md` | This file: the operating guide for AI agents working in the repo. |
| `CLAUDE.md` | Pointer that directs Claude Code to this file. |
| `.claude/skills/submission-review/SKILL.md` | Skill: review a contributed dataset for compliance, metadata, tier, and hours. |
| `.claude/skills/dataset-conversion/SKILL.md` | Skill: help a contributor convert source data into LeRobot v3.0. |

## Signal-Tier Vocabulary

Tier weights multiply contributed hours; higher tiers are rarer and more valuable.

- S1, native robot kinematics (joint and actuator state, motor commands, insertion depth, tip pose, teleoperation commands), weight x5. Example: OpenRC-style native capture.
- S2, tracked pose (electromagnetic tracking, fiber-optic or EM shape sensing, magnetic tracker through the tool channel), weight x4.
- S3, inferred pose (SLAM, SfM, or point tracking from camera; pose inferred from fluoroscopy), weight x2.
- S4, no kinematics, rich annotations instead (segmentation, depth or 3D, procedure phase, VQA, chain-of-thought traces, anomaly annotations, de-identified reports), weight x2.
- Raw, unlabeled video is NOT accepted.

## Collection-Setting Vocabulary

In preference order: clinical (human), in-vivo (animal), ex-vivo (animal tissue), phantom / bench-top, simulation (digital). Simulation is an explicitly welcomed contribution path. Per-setting hour minimums live in the RFP (`assets/open-h-endoluminal-rfp.pdf`, Section 3.6) and apply to RGB endoscopy datasets; fluoroscopy-based dataset proposals are considered case-by-case. LINK to the RFP for those numbers, never duplicate its tables (they would go stale).

## Feature-Naming Conventions

These names must be identical across the README, template, converters, and validator.

- Required features: `action` and `observation.state`. `observation.state` holds the PRIMARY proprioception describing the robot/endoscope state (native kinematics where available; the tracked pose only on platforms where that is the sole proprioceptive signal). Additional pose streams are auxiliary and live under `observation.meta.<field>` (e.g., `observation.meta.em_pose`) — never concatenated into `observation.state`.
- Camera streams: `observation.images.<view>` (endoluminal examples: `observation.images.endoscope`, `observation.images.fluoro`).
- Per-frame metadata: `observation.meta.<field>` (endoluminal examples: `observation.meta.scope_type`, `observation.meta.em_pose` if auxiliary).
- Camera-frame kinematics (REQUIRED best effort for RGB endoscopy): `observation.meta.camera_frame_delta_pose`, the per-step relative camera pose `[dx_m, dy_m, dz_m, dqx, dqy, dqz, dqw]` expressed in the previous frame's optical coordinates (first frame is the identity). The chip-on-tip camera is the end effector, so this is the endoluminal equivalent of the camera-frame end-effector pose used by rigid-arm datasets. Reference implementation: `absolute_poses_to_camera_frame_deltas()` in `scripts/conversion/hdf5_to_lerobot.py`. Fluoroscopy-only submissions are exempt.
- Timestep-level language: `instruction.text`.
- Camera intrinsics: stored as the calibration file `meta/calibration/camera_intrinsics.json` (keyed by camera feature name: OpenCV pinhole `fx`/`fy`/`cx`/`cy`, resolution, distortion model and coefficients). Strongly encouraged for RGB endoscopy. Reference writer: `write_camera_intrinsics()` in `scripts/conversion/hdf5_to_lerobot.py`.
- Ground-truth capture clocks: `observation.meta.host_stamp_ns` for the reference (video) stream, plus `observation.meta.<stream>_stamp_ns` (e.g., `observation.meta.kinematics_stamp_ns`, `observation.meta.tracker_stamp_ns`) for each additional stream captured at its own rate and resampled onto the frame timeline (all int64, Unix-epoch nanoseconds, one per frame). LeRobot's canonical `timestamp` column is always the frame timeline (`frame_index / fps`; the library does not accept explicit per-frame timestamps), so raw hardware stamps are preserved losslessly in these pass-through features. Streams must be captured or resampled at a fixed rate; when resampling to the frame timeline, choose the method by field type: linear interpolation for continuous or positional fields, spherical linear interpolation (Slerp) for quaternion orientation, and zeroth-order hold for categorical or slowly-changing metadata.
- Actions are positional setpoints (target positions and angles), not velocities.
- Quality bars (suggested): >= 20 Hz, >= 480p, MP4 video encoding, synchronization tolerance recorded via `tolerance_s` (typical 0.1 s).
- Flexible-robot state conventions: insertion depth, shaft rotation, tip bend angles (up-down, left-right) or tendon displacements; catheters similar, with guidewire advance/retract and rotate. Tip pose appears in `observation.state` only when it is the primary proprioception (e.g., an S2-only platform with no native kinematics, like the README's catheter example); otherwise it is auxiliary under `observation.meta.em_pose`.

## Common Workflows

### Review a submission

Use the `submission-review` skill (`.claude/skills/submission-review/SKILL.md`). In short: run `scripts/validation/validate_formatting.py` on the dataset, check every required metadata item against the RFP, compute hours of synchronized data, classify the actual signal tier from the features present, and produce a structured review with a verdict and concrete fixes.

### Help a contributor convert data

Use the `dataset-conversion` skill (`.claude/skills/dataset-conversion/SKILL.md`). In short: identify the source layout (HDF5, Zarr, ROS bags, CSV plus frames), establish synchronization with `scripts/synchronization/`, map streams to the feature-naming conventions above, adapt a converter from `scripts/conversion/`, then run the validator and iterate until it passes clean.

## Do Not Modify the RFP

The canonical, finalized RFP is `assets/open-h-endoluminal-rfp.pdf`; it must not be modified or replaced by agents. If other documents need numbers that live in the RFP (per-setting hour minimums, tier weights, the steering group table), link to the PDF rather than copying them.
