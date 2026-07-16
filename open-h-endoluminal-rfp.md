<!-- Working draft. Replace this file's contents with the finalized RFP when it lands. Source of truth during drafting: the shared Google Doc / Nigel's vault copy. -->

# Request for Proposals (RFP)

## Open-H-Endoluminal Collaborative Endoluminal and Interventional Robotics Dataset Initiative

## 1 Introduction

Open-H-Endoluminal is a collaborative dataset generation and collection effort led by NVIDIA, Johns Hopkins University, Houston Methodist Research Institute, Rice University, and the Technical University of Munich, together with partner institutions, focused on endoscopic, endoluminal, and interventional robotics: the flexible scopes, capsules, continuum and soft robots, and catheter-based systems that navigate the body's natural lumens and vasculature. Our goal is to align leading institutions to assemble at least 50,000 real and synthetic procedure episodes consisting of synchronized video, robot state and action, and language across these procedures, with an initial emphasis on lower-GI navigation. The resulting datasets will be used to train and evaluate vision-language-action models, world and simulation models, and core perception capabilities including depth estimation, 3D reconstruction, segmentation, and workflow understanding.

## 2 Steering Group

| Role | Name | Affiliation |
|---|---|---|
| Industry Lead | Dr. Mahdi Azizian | NVIDIA |
| Academic Co-Lead | Prof. Mathias Unberath | Johns Hopkins University |
| Academic Co-Lead | Prof. Farshid Alambeigi | Houston Methodist Research Institute / Rice University |
| Academic Co-Lead | Prof. Nassir Navab | Technical University of Munich |

## 3 Scope of Work

Participating teams may submit for one or more of the following domains. Data is accepted broadly across the endoluminal scope.

### 3.1 Lower and Upper GI Endoscopy

Tasks follow a shared hierarchy: navigation and intubation, screening and coverage, detection and diagnosis, and intervention (for example biopsy, polypectomy, and wound sealing). Platform examples: robotic flexible endoscopy systems, the OpenRC framework, and conventional flexible GI scopes.

### 3.2 Bronchoscopy

Navigation to airway targets and transbronchial biopsy.

### 3.3 Ureteroscopy and Transurethral

Navigation and diagnostic or interventional tasks in the urinary tract.

### 3.4 Capsule, Continuum, and Soft-Robotic Systems

Locomotion and navigation for capsule endoscopes and continuum or soft robots operating inside lumens.

### 3.5 Endovascular and Catheter-Based Intervention

Navigation to a vascular target, kept separate from the target intervention (for example clot removal). Because visualization is fluoroscopic rather than endoscopic, this domain may be supported through simulation.

**Out of scope:** laparoscopy, arthroscopy, and rigid-arm manipulation.

### 3.6 Data Requirements and Contribution Tiers

Each proposal must commit to a minimum data contribution and to providing time-aligned streams and required metadata.

**Action and state signal.** Submissions are graded into three signal tiers by the fidelity of their action and state signal. Label which you provide.

| Signal Tier | Action / State Signal | Examples | Signal Weight |
|---|---|---|---|
| S1 | Native robot kinematics | Joint and actuator state, motor commands, insertion depth, tip pose, teleoperation commands | ×20 |
| S2 | Tracked pose | Electromagnetic (EM) tracking, fiber-optic shape sensing, magnetic tracker through the tool channel | ×5 |
| S3 | Inferred pose | Pose inferred from camera (SLAM, SfM, point tracking) or from fluoroscopy | ×2 |

Video with rich labels (for example segmentation, depth or 3D, procedure phase, VQA, chain-of-thought, or polyp and lesion annotations) is considered on a case-by-case basis. Raw, unlabeled video is not accepted.

**Time-aligned streams.** Provide, synchronized: endoscopic video (RGB or RGB-D) and, where relevant, fluoroscopic or X-ray video; an action or state signal from the table above; and corresponding imaging where available, including CT and CT-fluoro pairs.

**Format.** Deliver data in the LeRobot data format. Where a robot is involved, supply calibration data and scope or robot CAD and kinematic-tree descriptions (USD, URDF, DH parameters, or equivalent).

**Required metadata (every submission).** Task intent (navigation, screening, detection, or intervention) and the target; device or platform; collection setting (in-vivo, ex-vivo, phantom, or simulation); modalities present with their synchronization and sample rates; and licence and de-identification status.

**Minimum contribution.** Minimums are specified in hours of synchronized data per collection setting. The S1 column is the base requirement; because S2 and S3 signals carry proportionally lower weight, their minimums scale up so that each tier represents an equivalent contribution. Preferred order of collection setting: clinical, in-vivo, ex-vivo, phantom or bench-top, simulation.

| Collection Setting | Definition | S1 (Native) | S2 (Tracked) | S3 (Inferred) |
|---|---|---|---|---|
| Clinical (Human) | Collected during approved human procedures with appropriate IRB or ethics oversight. | 2 h | 8 h | 20 h |
| In-Vivo (Animal) | Acquired on living tissue (for example a porcine model) under laboratory or veterinary supervision. | 3 h | 12 h | 30 h |
| Ex-Vivo (Animal Tissue) | Performed on isolated tissue samples (for example ex-vivo bowel or airway). | 5 h | 20 h | 50 h |
| Phantom / Bench-Top | Conducted on physical phantoms, synthetic tissues, or bench-top fixtures. | 8 h | 32 h | 80 h |
| Simulation (Digital) | Digitally simulated using physics-based or differentiable environments. | 10 h | 40 h | 100 h |

We encourage proposals to also provide: time-aligned narration or description of the sub-task (audio or text); correlated anonymized patient information where available; and labels for demonstration quality (expert, intermediate, novice) and task success (failure, recovery, success).

### 3.7 Focused Tracks and Alternative Formats

To capture the benefits of focused, high-quality datasets while keeping the initiative flexible, we invite groups to propose focused tracks around specific areas, for example lower-GI intubation on a shared reference phantom, polypectomy or biopsy through the tool channel, or endovascular navigation in simulation. We also welcome standardized pre-calibration and shared-phantom protocols and alternative data formats.

## 4 Timeline

| Milestone | Date |
|---|---|
| Private recruitment of leading groups | Underway |
| RFP released | September 2026 |
| Proposal submission deadline | October 2026 |
| Data collection window | October 2026 to January 2027 |
| Data cleanup and standardization | January 2027 |
| Model training and validation | February 2027 |
| Public release of dataset and models | March 2027 |

## 5 Eligibility and Partnership Models

- Academic institutions, start-ups, and industrial healthcare firms are welcome.
- International participation is encouraged.
- Consortia of multiple organisations may submit a joint proposal with a single lead.

**Collaboration benefits.** Selected partners will be named co-authors on the Open-H-Endoluminal dataset publication and on the follow-up paper describing the resulting models. Teams will receive early-access evaluation checkpoints of the models and the dataset itself for their own downstream experiments. Contributors without access to a robotic platform may participate through the simulation track. Participating teams will also receive recognition during the dataset and model releases at NVIDIA GTC 2027.

## 6 Proposal Requirements

Submissions (single PDF, ≤ 5 pages excluding appendices) must include:

1. Executive summary (≤ 500 words).
2. Task and domain selection: target domain or domains from Section 3, tasks along the hierarchy, device or platform, which signal tier (S1, S2, or S3) you provide, sensors, and committed hours of data per collection setting.
3. Technical approach: data-collection methodology, synchronization, quality assurance, and privacy safeguards (HIPAA and GDPR).
4. Team qualifications: key personnel biographies and prior related work.
5. Project plan: work-breakdown structure and Gantt chart aligned with the Section 4 timeline.
6. Data rights and IP statement: confirmation of intent to release data under the Creative Commons CC BY 4.0 licence (or compatible) and to comply with steering-group IP policy.

Submit proposals to shuver@nvidia.com with "Open-H-Endoluminal RFP" as the title.

## 7 Data Governance, Ethics, and Compliance

- **Patient privacy.** All human-subject data must be de-identified to HIPAA Safe Harbor standards or equivalent (GDPR for European contributors), including scrubbing DICOM and imaging metadata on CT, fluoroscopy, and video exports.
- **Regulatory.** Proposers are responsible for obtaining IRB or ethics approval where required.
- **Licensing.** Final datasets will be released under CC BY 4.0. Synthetic assets must be free of third-party IP encumbrances.

## 8 Contact Information

- **Technical questions:** Nigel Nelson, nigeln@nvidia.com (NVIDIA)
- **Administrative questions:** Sean Huver, shuver@nvidia.com
