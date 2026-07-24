---
name: submission-review
description: Review an Open-H-Endoluminal dataset submission for format compliance, required metadata, signal tier, and hours accounting. Use when asked to review, validate, or grade a contributed dataset.
---

# Submission Review

Review a contributed dataset against Open-H-Endoluminal requirements. Submissions are LeRobot format v3.0 datasets (the guide pins the `lerobot[dataset]==0.6.0` package; any lerobot >= 0.4.0 can read format v3.0 — the package version and the dataset format version are separate versioning schemes). Contributions are measured in hours of synchronized data, never in trajectories or episode counts.

## Procedure

### 1. Run the validator

```bash
python scripts/validation/validate_formatting.py <dataset_path> --verbose
```

Capture the full report. Every ERROR must be resolved before acceptance; WARNINGs become fix items in the review.

### 2. Verify the LeRobot v3.0 layout and the dataset README

Expected layout:

- `meta/info.json` with `"codebase_version": "v3.0"`
- `meta/stats.json`
- `meta/tasks.parquet`
- `meta/episodes/chunk-*/file-*.parquet`
- `data/chunk-*/file-*.parquet` (each parquet aggregates MULTIPLE episodes)
- `videos/<camera_key>/chunk-*/file-*.mp4`

If you see `meta/episodes.jsonl`, `meta/episodes_stats.jsonl`, `meta/tasks.jsonl`, or `data/chunk-*/episode_*.parquet`, the dataset is still v2.1: the contributor must convert with the official script (module `lerobot.scripts.convert_dataset_v21_to_v30` in lerobot 0.6.0; for local-only datasets add `--root <dataset_dir> --push-to-hub false`, where `--root` is the dataset directory itself).

Confirm `meta/README.md` exists and is a completed copy of `templates/dataset_template.md`: no unfilled `[...]` placeholders, and the synchronization section documents the method and sample rates.

### 3. Check required metadata

Every submission must state, in `meta/README.md` and the dataset metadata:

- Task intent + target (navigation, screening/coverage, detection/diagnosis, or intervention, with the anatomical target).
- Device or platform.
- Collection setting (clinical, in-vivo, ex-vivo, phantom / bench-top, or simulation).
- Modalities present, with synchronization method and sample rates.
- Claimed signal tier (S1 to S4).
- Licence (CC BY 4.0) and de-identification status (HIPAA Safe Harbor or equivalent, GDPR for European contributors).
- Where a robot is involved: calibration data and scope or robot CAD / kinematic-tree descriptions (USD, URDF, DH parameters, or equivalent).
- RGB endoscopy submissions: `observation.meta.camera_frame_delta_pose` present (best-effort requirement, with the supporting calibration and derivation method documented), or a justification in `meta/README.md` for why camera-frame kinematics were infeasible. Fluoroscopy-only submissions are exempt.
- Camera intrinsics (strongly encouraged for RGB endoscopy): a `meta/calibration/camera_intrinsics.json` file keyed by camera feature name. Note its absence as a fix item rather than a hard blocker.

### 4. Classify the actual signal tier

Inspect the `features` in `meta/info.json` and sample the data, then classify:

- S1 (weight x5): native robot kinematics in `action` / `observation.state` (joint and actuator state, motor commands, insertion depth, tip pose, teleoperation commands).
- S2 (weight x4): tracked pose (electromagnetic tracking, fiber-optic or EM shape sensing, magnetic tracker through the tool channel, e.g. `observation.meta.em_pose`).
- S3 (weight x2): inferred pose (SLAM, SfM, or point tracking from camera; pose inferred from fluoroscopy).
- S4 (weight x2): no kinematics, rich annotations instead (segmentation, depth or 3D, procedure phase, VQA, chain-of-thought traces, anomaly annotations, de-identified reports).

Flag any mismatch between the claimed tier and what the features actually contain (for example, `observation.state` filled with zeros or a constant is not S1).

### 5. Compute hours and compare against minimums

Hours = `total_frames / fps / 3600`, both read from `meta/info.json`. Cross-check against the total hours stated in `meta/README.md`. Compare against the per-setting hour minimums in the RFP, `assets/open-h-endoluminal-rfp.pdf` (Section 3.6); read them from the RFP, never from memory, and never copy the RFP tables into other documents. Note the RFP minimums table (S1 through S4) applies only to RGB endoscopy datasets; fluoroscopy-based submissions are considered case-by-case, so flag their hours for steering review instead of a hard pass/fail. Report hours as well as episode counts.

### 6. Confirm this is not raw unlabeled video

Raw, unlabeled video is not accepted. If the dataset contains only `observation.images.*` streams with no state, action, or pose signals and no rich labels, it does not qualify. Navigation data must show movement toward a stated target (in the per-episode task text and `instruction.text` where present), never unlabeled wandering.

### 7. Produce the review

Verdict is one of: **accept**, **accept-with-fixes**, **needs-work**.

- accept: validator clean, all metadata present, tier and hours verified.
- accept-with-fixes: no structural blockers, but specific items must be corrected (list them with exact instructions).
- needs-work: validator errors, missing required metadata, tier misclassification, raw unlabeled video, or hours below the applicable minimum.

## Reporting format

```markdown
# Submission Review: <dataset name>

- Dataset path: <path>
- Reviewed: <date>
- Verdict: accept | accept-with-fixes | needs-work

## Summary
- Hours (computed): <X.X> h  (claimed: <Y.Y> h)
- Signal tier (assessed): S<i>  (claimed: S<j>)
- Collection setting: <setting>; RFP minimum for this setting/tier: see assets/open-h-endoluminal-rfp.pdf Section 3.6
- Validator: <E> errors, <W> warnings

## Findings
| # | Check | Status | Details | Required fix |
|---|-------|--------|---------|--------------|
| 1 | Validator | pass/fail | ... | ... |
| 2 | v3.0 layout + meta/README.md | pass/fail | ... | ... |
| 3 | Required metadata | pass/fail | ... | ... |
| 4 | Signal tier | pass/mismatch | ... | ... |
| 5 | Hours vs minimum | pass/fail | ... | ... |
| 6 | Not raw unlabeled video | pass/fail | ... | ... |

## Fix instructions
1. <concrete, ordered steps the contributor should take>
```

Style notes for the review text: no em dashes; initiative names hyphenated (Open-H-Endoluminal); hours, never trajectory or episode counts, as the measure of contribution. Technical questions: Nigel Nelson, nigeln@nvidia.com.
