<!--
Open-H-Endoluminal Dataset README Template (v1.0)
Please fill out this template and place the completed file at meta/README.md inside your
LeRobot dataset (dataset format v3.0). This file helps others understand the context and
details of your contribution.
-->

# [Dataset Name] - README

---

## 📋 At a Glance

*Provide a one-sentence summary of your dataset.*

**Example:** *Teleoperated colonoscope navigation to the cecum on a silicone colon phantom using an OpenRC-style robotic colonoscopy kit, with native motor kinematics (S1).*

---

## 📖 Dataset Overview

*Briefly describe the purpose and content of this dataset. What key skills or scenarios does it demonstrate? Report scale in hours of synchronized data first; episode and frame counts are secondary detail.*

**Example:** *This dataset contains 42 hours of synchronized endoscope video and native robot kinematics from expert endoscopists teleoperating a robotic colonoscope to navigate from insertion to the cecum on three phantom models. It includes successful runs, failures, and recovery attempts to provide a robust dataset for training imitation learning policies.*

| | |
| :--- | :--- |
| **Total Hours of Synchronized Data** (REQUIRED) | `[Number]` *(if collection settings are mixed, break hours down per setting)* |
| **Hours per Setting** (REQUIRED if mixed) | `[e.g., Phantom: 30 h, Simulation: 12 h]` |
| **Episodes** | `[Number]` |
| **Frames** | `[Number]` |
| **Dataset Format** | LeRobot dataset format v3.0 |
| **License** (REQUIRED) | CC BY 4.0 *(default; state and justify any deviation)* |
| **De-identification Status** (REQUIRED) | `[e.g., De-identified to HIPAA Safe Harbor / GDPR-equivalent / N/A (no human-subject data)]` |
| **Version** | `[e.g., 1.0]` |

---

## 🎯 Tasks & Domain

### Domain (REQUIRED)

*Select the primary domain(s) for this dataset.*

- [ ] **Lower GI endoscopy**
- [ ] **Upper GI endoscopy**
- [ ] **Bronchoscopy**
- [ ] **Endovascular / catheter-based**
- [ ] **Ureteroscopy / transurethral**
- [ ] **Capsule endoscopy**
- [ ] **Continuum / soft robot**
- [ ] **Other endoluminal** (Please specify: `[Your Domain]`)

### Task Intent (REQUIRED)

*Check every task category demonstrated in this dataset.*

- [ ] **Navigation**
- [ ] **Screening / Coverage**
- [ ] **Detection / Diagnosis**
- [ ] **Intervention** (e.g., biopsy, polypectomy, clot removal)

**Target description (REQUIRED):** *Describe the stated target of each task in free text. Navigation data must show movement toward a stated target, never unlabeled wandering.*

**Example:** *Navigation from rectal insertion to cecal intubation, confirmed by appendiceal orifice visualization; a subset of episodes continues to cold-snare polypectomy of simulated polyps in the ascending colon.*

### Demonstrated Skills

*List the primary skills or procedures demonstrated in this dataset.*

***Example:***
- Lumen centering and advancement
- Loop reduction
- Retroflexion
- ...

---

## 📡 Signal Tier (REQUIRED)

*Check the highest tier present, plus any additional tiers included. See the Open-H-Endoluminal RFP (`open-h-endoluminal-rfp.md` in the contribution guide repository) for how tiers weight contributed hours; its tables cover S1 to S3, while S4 submissions are weighted x1 and reviewed case-by-case. Raw, unlabeled video is not accepted.*

- [ ] **S1: Native robot kinematics** (joint and actuator state, motor commands, insertion depth, tip pose, teleoperation commands)
- [ ] **S2: Tracked pose** (electromagnetic tracking, fiber-optic shape sensing, magnetic tracker through the tool channel)
- [ ] **S3: Inferred pose** (SLAM, SfM, or point tracking from camera; pose inferred from fluoroscopy)
- [ ] **S4: Video with rich labels** (segmentation, depth or 3D, procedure phase, VQA, chain-of-thought traces, polyp and lesion annotations)

**Signal source (REQUIRED):** *One line describing where the tier-defining signal comes from.*

**Example:** *Tip pose from a 6-DoF electromagnetic sensor threaded through the working channel, sampled at 40 Hz by the tracking workstation.*

---

## 🔬 Data Collection Details

### Collection Setting (REQUIRED)

*Check all settings present. If mixed, report hours per setting in the Dataset Overview table. Per-setting hour minimums are listed in the Open-H-Endoluminal RFP (`open-h-endoluminal-rfp.md` in the contribution guide repository).*

- [ ] **Clinical (human)**
- [ ] **In-vivo (animal)**
- [ ] **Ex-vivo (animal tissue)**
- [ ] **Phantom / bench-top**
- [ ] **Simulation (digital)**

### Collection Method

*How was the data collected?*

- [ ] **Human Teleoperation** (robotic platform driven by an operator)
- [ ] **Manual Procedure** (hand-held scope or catheter, instrumented for recording)
- [ ] **Programmatic / State-Machine**
- [ ] **AI Policy / Autonomous**
- [ ] **Other** (Please specify: `[Your Method]`)

### Operator Details

| | Description |
| :--- | :--- |
| **Operator Count** | `[Number of unique people who collected data]` |
| **Operator Skill Level** | `[ ] Expert (e.g., attending endoscopist, interventionalist)` <br> `[ ] Intermediate (e.g., fellow, trained researcher)` <br> `[ ] Novice (e.g., ML researcher with minimal experience)` <br> `[ ] N/A` |
| **Per-Episode Skill Labels Included?** | `[ ] Yes  [ ] No` *(encouraged)* |
| **Collection Period** | From `[YYYY-MM-DD]` to `[YYYY-MM-DD]` |

### Recovery Demonstrations

*Does this dataset include examples of recovering from failure (e.g., loop resolution, re-finding the lumen after red-out, re-crossing a vessel branch)?*

- [ ] **Yes**
- [ ] **No**

**If yes, please briefly describe the recovery process:**

*Example: In 6 hours of episodes, demonstrations are initialized from a lost-lumen state; the operator withdraws, re-centers the lumen, and resumes advancement toward the cecum.*

*If you include recovery or failure episodes, we encourage recording them as optional first-class `recovery` and `failure` splits, stored as episode-index ranges in `meta/info.json` alongside the standard train/val/test splits.*

### De-identification & Ethics (REQUIRED)

| | |
| :--- | :--- |
| **Human-subject data present?** | `[ ] Yes  [ ] No` |
| **Standard met** | `[ ] HIPAA Safe Harbor  [ ] GDPR-equivalent  [ ] N/A (no human-subject data)` |
| **Method** | `[e.g., out-of-body frame removal, DICOM/metadata scrubbing, burned-in text redaction, voice removal from narration]` |
| **IRB / ethics approval** | `[Protocol ID or N/A; proposers are responsible for IRB/ethics approval]` |

---

## 💡 Diversity Dimensions

*Check all dimensions that were intentionally varied during data collection.*

- [ ] **Anatomy / Subject** (e.g., different patients, animals, or phantom models)
- [ ] **Scope / Catheter / Robot Embodiment** (if multiple devices or platforms were used)
- [ ] **Route / Target** (e.g., different airway segments, vessel branches, colon segments)
- [ ] **Lighting / Insufflation / Contrast Conditions**
- [ ] **Lesion or Pathology Variation** (e.g., polyp size, morphology, location)
- [ ] **Task Execution** (e.g., different techniques for the same task)
- [ ] **Operator** (multiple operators or skill levels)
- [ ] **Other** (Please specify: `[Your Dimension]`)

*If you checked any of the above please briefly elaborate below.*

**Example:** *We rotated across three colon phantom models with distinct loop geometries, and every 5 hours we swapped the simulated polyp set across three size classes (diminutive, small, large) placed at randomized positions in the ascending and transverse colon.*

---

## 🛠️ Equipment & Setup

### Platform(s)

*List the primary scope, catheter, robot, or simulator used. Generic descriptions are fine if the device cannot be named.*

- **Platform 1:** `[e.g., OpenRC robotic colonoscopy research kit (arXiv:2604.03781)]`
- **Platform 2:** `[e.g., clinical flexible colonoscope, 13.2 mm shaft, hand-held]`

### Sensors & Cameras

*List every sensor and camera with its sample rate and how it is synchronized. Suggested quality bars: >= 20 Hz for control and pose streams, >= 480p MP4 for video. (Add and remove rows as needed.)*

| Type | Model/Details | Sample Rate | Sync Method |
| :--- | :--- | :--- | :--- |
| **Endoscope Camera (chip-on-tip)** | `[e.g., 1920x1080 chip-on-tip sensor]` | `[e.g., 30 fps]` | `[e.g., hardware trigger, host timestamp]` |
| **Fluoroscopy** | `[e.g., C-arm, 960x960]` | `[e.g., 15 fps]` | `[e.g., frame-grabber host timestamp]` |
| **EM Tracker** | `[e.g., 6-DoF sensor in working channel]` | `[e.g., 40 Hz]` | `[e.g., vendor SDK timestamp mapped to host clock]` |
| **Shape-Sensing Fiber** | `[e.g., fiber Bragg grating array along shaft]` | `[e.g., 100 Hz]` | `[e.g., interrogator clock, offset-calibrated]` |
| **Motor Encoders / Joint State** | `[e.g., insertion, rotation, and tendon actuator encoders]` | `[e.g., 100 Hz]` | `[e.g., ROS 2 header.stamp, shared clock]` |
| **Other** | `[Specify]` | `[Rate]` | `[Method]` |

---

## 🎯 Action & State Space Representation

*Describe how actions and device states are represented in your dataset. This is crucial for understanding data compatibility and enabling effective policy learning. Use the standard feature names: `action`, `observation.state`, camera streams as `observation.images.<view>` (e.g., `observation.images.endoscope`, `observation.images.fluoro`), per-frame metadata as `observation.meta.<field>`, and timestep-level language in `instruction.text`. Actions should be positional (position or angle setpoints), not velocities.*

### Action Space Representation

**Primary Action Representation:**
- [ ] **Flexible-Scope Convention** (insertion/retraction, shaft rotation, tip bend angles or tendon displacements)
- [ ] **Catheter Convention** (guidewire advance/retract and rotate; catheter advance and rotate)
- [ ] **Joint / Actuator Space** (direct joint or motor commands, for robotic platforms)
- [ ] **Tip Pose** (absolute or relative pose commands)
- [ ] **Other** (Please specify: `[Your Representation]`)

**Orientation Representation (for any pose components):**
- [ ] **Quaternions** (x, y, z, w)
- [ ] **Euler Angles** (roll, pitch, yaw)
- [ ] **Axis-Angle** (rotation vector)
- [ ] **Rotation Matrix** (3x3 matrix)
- [ ] **Other** (Please specify: `[Your Representation]`)

**Action Dimensions (REQUIRED):**
*List every action dimension with its meaning, reference frame, and units.*

**Example:**
```
action: [insertion_pos, rotation_pos, bend_ud, bend_lr]
- insertion_pos: commanded insertion position along the scope axis (mm), measured from the anal verge, positive = deeper
- rotation_pos: commanded shaft rotation angle (rad), cumulative, positive = clockwise viewed from the operator
- bend_ud: commanded tip bend angle in the up-down plane (rad), tip frame
- bend_lr: commanded tip bend angle in the left-right plane (rad), tip frame
```

### State Space Representation

**State Information Included:**
- [ ] **Insertion Depth** (measured at the insertion point)
- [ ] **Shaft Rotation**
- [ ] **Tip Bend Angles or Tendon Displacements**
- [ ] **Tip Pose** (from EM tracking, shape sensing, or inference — in `observation.state` only if it is the platform's primary proprioception; otherwise auxiliary under `observation.meta.<field>`, e.g. `observation.meta.em_pose`)
- [ ] **Camera-Frame Delta Pose** (`observation.meta.camera_frame_delta_pose`; see the Camera-Frame Kinematics section below)
- [ ] **Joint / Motor Positions and Velocities** (robotic platforms)
- [ ] **Guidewire / Catheter State** (advance, rotation)
- [ ] **Force / Torque or Contact Readings**
- [ ] **Other** (Please specify: `[Your State Info]`)

**State Dimensions (REQUIRED):**
*List every state dimension with its meaning, reference frame, and units.*

**Example:**
```
observation.state: [ins_depth, shaft_rot, bend_ud, bend_lr]
- ins_depth: insertion depth at the anal verge (mm)
- shaft_rot: cumulative shaft rotation (rad), positive = clockwise viewed from the operator
- bend_ud, bend_lr: measured tip bend angles (rad), tip frame

observation.meta.em_pose: [tip_x, tip_y, tip_z, tip_qx, tip_qy, tip_qz, tip_qw]
- auxiliary EM-tracked tip pose in the field-generator frame (mm, unit quaternion).
  observation.state carries the platform's PRIMARY proprioception (here the native
  kinematics); an additional tracked pose is documented as its own
  observation.meta.<field> feature rather than concatenated into the state. (On a
  platform whose only proprioception is the tracked pose, the pose itself is
  observation.state instead.)
```

### Camera-Frame Kinematics (REQUIRED best effort for RGB endoscopy)

*In endoluminal robotics the chip-on-tip camera is the end effector, so camera-frame motion is the equivalent of the camera-frame end-effector pose used by rigid-arm datasets such as Open-X Embodiment. Provide the feature `observation.meta.camera_frame_delta_pose = [dx_m, dy_m, dz_m, dqx, dqy, dqz, dqw]`: the relative pose of the camera from the previous frame to the current frame, expressed in the previous frame's optical coordinates (OpenCV convention: +x right, +y down, +z along the optical axis). The first frame of each episode is the identity `[0, 0, 0, 0, 0, 0, 1]`. This is a best-effort requirement for RGB endoscopy submissions; fluoroscopy-only datasets are exempt. A reference implementation is provided as `absolute_poses_to_camera_frame_deltas()` in `scripts/conversion/hdf5_to_lerobot.py` in the contribution guide repository.*

**Camera-frame kinematics provided?**
- [ ] **Yes**, as `observation.meta.camera_frame_delta_pose`
- [ ] **No** (justify below why it was infeasible)

**Derivation method:**
- [ ] **From native kinematics (S1)** (forward kinematics plus tip-to-camera hand-eye calibration)
- [ ] **From tracked pose (S2)** (EM or shape-sensing pose plus sensor-to-camera calibration)
- [ ] **From inferred pose (S3)** (monocular or stereo SLAM / visual odometry ego-motion)
- [ ] **Other / not provided** (Please specify: `[Your Method]`)

**Scale:**
- [ ] **Metric (meters)**
- [ ] **Normalized / up-to-scale** (state the normalization; typical for monocular SLAM)

*Describe where the supporting calibration lives in the dataset and any known error characteristics:* `[e.g., meta/calibration/hand_eye.json contains the tip-to-camera transform; translation error is below 2 mm RMS against EM tracking on bench-top trajectories]`

*If camera-frame kinematics are not provided, justify why deriving them was infeasible:* `[e.g., hand-held clinical scope with no kinematic or pose sensing, and monocular visual odometry failed on the majority of frames due to red-out and specular reflections]`

### Calibration & Kinematic Descriptions (REQUIRED where a robot is involved)

- [ ] **Calibration data provided** (e.g., camera intrinsics and extrinsics, EM-to-camera or hand-eye transforms)
- [ ] **CAD / kinematic-tree description provided** (USD, URDF, DH parameters, or equivalent for the scope, catheter, or robot)

*Describe what is included and where it lives in the dataset:* `[e.g., meta/calibration/ contains camera intrinsics and the EM-to-camera transform; a URDF of the actuation unit is included]`

---

## ⏱️ Data Synchronization Approach (REQUIRED)

*Describe how you achieved temporal alignment across sensors, cameras, and the device or robot. Document the synchronization method and the sample rate of every modality, the provenance of your timestamps (which clock stamped each stream), any measured inter-sensor skew, and the `tolerance_s` value written into your LeRobot dataset (typical: 0.1 s). Note that the dataset's canonical `timestamp` column is the frame timeline (`frame_index / fps`); if you preserved your raw hardware clocks as `observation.meta.host_stamp_ns` (encouraged — int64 Unix-epoch nanoseconds), say so here.*

**Example:** *We collect motor encoder states from an OpenRC-style actuation unit and chip-on-tip endoscope frames at 100 Hz and 30 fps respectively, all running in ROS 2 on the same workstation clocked with ROS Time. Both drivers stamp outgoing messages' header.stamp fields from the shared system clock, and we record the joint-state and image topics in a single rosbag2 session. During export to LeRobot v3.0, streams are aligned to the 30 fps frame timeline (the dataset's canonical `timestamp` column is frame_index / 30), each row's header.stamp is preserved verbatim as `observation.meta.host_stamp_ns`, and tolerance_s is set to 0.1. Offline checks show inter-sensor skew stays below 3 ms across a 20-minute capture.*

---

## 👥 Attribution & Contact

*Please provide attribution for the dataset creators and a point of contact.*

| | |
| :--- | :--- |
| **Dataset Lead** | `[Name1, Name2, ...]` |
| **Institution** | `[Your Institution]` |
| **Contact Email** | `[email1@example.com, email2@example.com, ...]` |
| **Citation (BibTeX)** | <pre><code>@misc{[your_dataset_name_2026],<br>  author = {[Your Name(s)]},<br>  title = {[Your Dataset Title]},<br>  year = {2026},<br>  publisher = {Open-H-Endoluminal},<br>  note = {De-identified to HIPAA Safe Harbor or an equivalent standard prior to release.}<br>}</code></pre> |

*Questions about this template? Technical: Nigel Nelson (nigeln@nvidia.com). Administrative: Sean Huver (shuver@nvidia.com). Community: https://discord.gg/YZEhNcTHtc*
