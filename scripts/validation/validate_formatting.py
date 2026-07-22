#!/usr/bin/env python
"""
Open-H-Endoluminal Dataset Validation Script

This script validates LeRobot datasets for compliance with:
1. LeRobot dataset format v3.0 specifications
2. Open-H-Endoluminal data collection initiative requirements and
   recommendations

This is a LOCAL validation tool designed to be run as a final check before
uploading your dataset. It does NOT require internet access or authentication.

Most checks (structure, metadata, timestamps, video) run with only
pyarrow/pandas, numpy, and OpenCV installed. The final load check uses the
lerobot package (v0.4.0 or later, which reads dataset format v3.0) and
degrades to a warning when lerobot is not installed. Note that the lerobot
PACKAGE version (0.4.0+) and the dataset FORMAT version (v3.0) are separate
versioning schemes.

Usage:
    python scripts/validation/validate_formatting.py /path/to/lerobot/dataset

    For verbose output:
    python scripts/validation/validate_formatting.py /path/to/dataset --verbose

The script performs comprehensive checks on:
- Dataset structure and format compliance (LeRobot v3.0 layout)
- Required features and naming conventions
- The raw-unlabeled-video rejection rule (a state, action, or pose signal,
  or S4-qualifying rich labels, must accompany every video stream)
- Best-effort camera-frame kinematics for RGB endoscopy submissions
  (observation.meta.camera_frame_delta_pose)
- meta/README.md documentation (synchronization method, signal tier)
- Hours accounting (hours of synchronized data are the contribution unit)
- Data quality metrics (fps, resolution, synchronization)
- Timestamp integrity
- Recovery/failure split handling
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Suppress FFmpeg/AV1 warnings
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("FFMPEG_HIDE_BANNER", "1")

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is expected but not mandatory
    cv2 = None

# NOTE: lerobot is intentionally NOT imported at module level. All structural,
# metadata, timestamp, and video checks run without it; only the final load
# check imports lerobot, and it degrades to a WARNING when the package is
# absent.

MIN_LEROBOT_VERSION = "0.4.0"

V21_CONVERSION_HINT = (
    "If your dataset was collected in LeRobot format v2.1, convert it with the "
    "official script that ships with LeRobot 0.6.0: "
    "python -m lerobot.scripts.convert_dataset_v21_to_v30 "
    "(for local-only datasets add --root <dataset_dir> --push-to-hub false, "
    "where --root is the dataset directory itself)"
)


def _parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse a version string into a comparable (major, minor, patch) tuple."""
    parts: List[int] = []
    for chunk in str(version_str).split(".")[:3]:
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


class ValidationLevel(Enum):
    """Validation severity levels"""

    ERROR = "ERROR"  # Must fix for compliance
    WARNING = "WARNING"  # Should fix for best practices
    INFO = "INFO"  # Suggestions for improvement
    SUCCESS = "SUCCESS"  # Validation passed


@dataclass
class ValidationResult:
    """Container for validation results"""

    level: ValidationLevel
    category: str
    message: str
    details: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report"""

    results: List[ValidationResult] = field(default_factory=list)
    dataset_path: Optional[Path] = None
    # Hours of synchronized data are the Open-H-Endoluminal contribution unit.
    total_hours: Optional[float] = None  # from info.json total_frames / fps
    episode_hours: Optional[float] = None  # from per-episode length sum / fps

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.INFO)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.SUCCESS)

    @property
    def is_compliant(self) -> bool:
        return self.error_count == 0


class OpenHEndoluminalValidator:
    """Validator for Open-H-Endoluminal LeRobot datasets (format v3.0)"""

    # Required LeRobot v3.0 structure
    REQUIRED_DIRS = ["data", "videos", "meta"]
    REQUIRED_METADATA_FILES = [
        "info.json",
        "stats.json",
        "tasks.parquet",
    ]
    # Legacy v2.1 metadata files that must NOT be present in a v3.0 dataset
    LEGACY_V21_METADATA_FILES = [
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "tasks.jsonl",
    ]

    # Required features based on Open-H-Endoluminal guidelines
    REQUIRED_FEATURES = ["action", "observation.state"]
    RECOMMENDED_IMAGE_PREFIX = "observation.images."

    # Open-H-Endoluminal specific requirements
    MIN_FPS = 20  # Minimum recommended FPS
    MIN_RESOLUTION = (480, 480)  # Minimum recommended resolution (height, width)

    # Signal-tier keywords for the meta/README.md documentation check
    SIGNAL_TIER_REGEX = re.compile(r"\bS[1-4]\b")
    SIGNAL_TIER_KEYWORDS = [
        "kinematics",
        "tracked pose",
        "inferred pose",
        "rich labels",
    ]

    def __init__(self, dataset_path: Path, verbose: bool = False):
        """
        Initialize validator with dataset path

        Args:
            dataset_path: Path to local dataset directory
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.report = ValidationReport()

        self.dataset_path = Path(dataset_path)
        self.report.dataset_path = self.dataset_path

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_result(
        self,
        level: ValidationLevel,
        category: str,
        message: str,
        details: Optional[str] = None,
    ):
        """Add a validation result to the report"""
        result = ValidationResult(level, category, message, details)
        self.report.results.append(result)

        if self.verbose or level in [ValidationLevel.ERROR, ValidationLevel.WARNING]:
            self._print_result(result)

    def _print_result(self, result: ValidationResult):
        """Print a validation result with formatting"""
        symbols = {
            ValidationLevel.ERROR: "❌",
            ValidationLevel.WARNING: "⚠️",
            ValidationLevel.INFO: "ℹ️",
            ValidationLevel.SUCCESS: "✅",
        }
        colors = {
            ValidationLevel.ERROR: "\033[91m",
            ValidationLevel.WARNING: "\033[93m",
            ValidationLevel.INFO: "\033[94m",
            ValidationLevel.SUCCESS: "\033[92m",
        }
        reset = "\033[0m"

        symbol = symbols[result.level]
        color = colors[result.level]

        print(
            f"{color}{symbol} [{result.level.value}] {result.category}: {result.message}{reset}"
        )
        if result.details and self.verbose:
            print(f"    Details: {result.details}")

    @staticmethod
    def _get_pandas():
        """Import pandas lazily so structural checks work without it."""
        try:
            import pandas as pd

            return pd
        except ImportError:
            return None

    def _load_info(self) -> Optional[Dict]:
        """Load meta/info.json, returning None on any failure."""
        info_path = self.dataset_path / "meta" / "info.json"
        if not info_path.exists():
            return None
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _data_parquet_files(self) -> List[Path]:
        """List v3.0 data files: data/chunk-*/file-*.parquet."""
        data_dir = self.dataset_path / "data"
        files: List[Path] = []
        if data_dir.exists():
            for chunk_dir in sorted(data_dir.glob("chunk-*")):
                files.extend(sorted(chunk_dir.glob("file-*.parquet")))
        return files

    def _episode_metadata_files(self) -> List[Path]:
        """List v3.0 episode metadata files: meta/episodes/chunk-*/file-*.parquet."""
        episodes_dir = self.dataset_path / "meta" / "episodes"
        files: List[Path] = []
        if episodes_dir.exists():
            for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
                files.extend(sorted(chunk_dir.glob("file-*.parquet")))
        return files

    # ------------------------------------------------------------------
    # Structure checks (LeRobot v3.0 layout, no lerobot import required)
    # ------------------------------------------------------------------

    def validate_directory_structure(self):
        """Validate LeRobot v3.0 directory structure"""
        category = "Directory Structure"

        # Check required directories
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Required directory '{dir_name}' not found",
                )
            else:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"Required directory '{dir_name}' exists",
                )

        # Check for data directory with parquet chunks
        data_dir = self.dataset_path / "data"
        if data_dir.exists():
            chunk_dirs = sorted(data_dir.glob("chunk-*"))
            if not chunk_dirs:
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    "No chunk directories found in data/ (expected format: data/chunk-000/)",
                )
            else:
                parquet_files = self._data_parquet_files()

                # Detect leftover v2.1 per-episode files
                legacy_files = []
                for chunk_dir in chunk_dirs:
                    legacy_files.extend(chunk_dir.glob("episode_*.parquet"))
                if legacy_files:
                    self.add_result(
                        ValidationLevel.ERROR,
                        category,
                        f"Found {len(legacy_files)} legacy v2.1 per-episode parquet file(s) "
                        "(episode_*.parquet). Format v3.0 aggregates multiple episodes "
                        "per file as data/chunk-*/file-*.parquet",
                        V21_CONVERSION_HINT,
                    )

                if not parquet_files:
                    self.add_result(
                        ValidationLevel.ERROR,
                        category,
                        "No data parquet files found "
                        "(expected format: data/chunk-*/file-*.parquet, "
                        "each file aggregating multiple episodes)",
                        V21_CONVERSION_HINT,
                    )
                else:
                    self.add_result(
                        ValidationLevel.SUCCESS,
                        category,
                        f"Found {len(parquet_files)} data parquet file(s) in "
                        f"{len(chunk_dirs)} chunk(s)",
                    )

        # Check videos directory layout: videos/<camera_key>/chunk-*/file-*.mp4
        videos_dir = self.dataset_path / "videos"
        if videos_dir.exists():
            camera_dirs = sorted(p for p in videos_dir.iterdir() if p.is_dir())
            if not camera_dirs:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    "No camera directories found in videos/ "
                    "(expected format: videos/<camera_key>/chunk-*/file-*.mp4)",
                )
            else:
                well_formed = 0
                for camera_dir in camera_dirs:
                    mp4s = list(camera_dir.glob("chunk-*/file-*.mp4"))
                    if mp4s:
                        well_formed += 1
                    else:
                        self.add_result(
                            ValidationLevel.WARNING,
                            category,
                            f"Camera directory 'videos/{camera_dir.name}/' contains no "
                            "chunk-*/file-*.mp4 files (v3.0 layout: "
                            "videos/<camera_key>/chunk-*/file-*.mp4)",
                        )
                    if not camera_dir.name.startswith(self.RECOMMENDED_IMAGE_PREFIX):
                        self.add_result(
                            ValidationLevel.WARNING,
                            category,
                            f"Camera key '{camera_dir.name}' does not use the "
                            f"'{self.RECOMMENDED_IMAGE_PREFIX}' prefix",
                            "Use names like observation.images.endoscope or "
                            "observation.images.fluoro",
                        )
                if well_formed:
                    self.add_result(
                        ValidationLevel.SUCCESS,
                        category,
                        f"Found {well_formed} camera director(ies) with v3.0 video layout",
                    )

    def validate_metadata_files(self):
        """Validate required v3.0 metadata files"""
        category = "Metadata Files"
        metadata_dir = self.dataset_path / "meta"

        if not metadata_dir.exists():
            self.add_result(
                ValidationLevel.ERROR, category, "Metadata directory not found"
            )
            return

        # Check required metadata files
        for file_name in self.REQUIRED_METADATA_FILES:
            file_path = metadata_dir / file_name
            if not file_path.exists():
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Required metadata file '{file_name}' not found",
                    V21_CONVERSION_HINT,
                )
            else:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"Metadata file '{file_name}' exists",
                )

        # Check the v3.0 episodes metadata directory
        episode_files = self._episode_metadata_files()
        if not (metadata_dir / "episodes").exists():
            self.add_result(
                ValidationLevel.ERROR,
                category,
                "Required metadata directory 'meta/episodes/' not found "
                "(expected meta/episodes/chunk-*/file-*.parquet)",
                V21_CONVERSION_HINT,
            )
        elif not episode_files:
            self.add_result(
                ValidationLevel.ERROR,
                category,
                "No episode metadata parquet files found under meta/episodes/ "
                "(expected meta/episodes/chunk-*/file-*.parquet)",
            )
        else:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"Found {len(episode_files)} episode metadata parquet file(s) "
                "under meta/episodes/",
            )

        # Detect leftover v2.1 metadata files
        for legacy_name in self.LEGACY_V21_METADATA_FILES:
            if (metadata_dir / legacy_name).exists():
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Legacy v2.1 metadata file '{legacy_name}' found. "
                    "This file does not exist in format v3.0 "
                    "(replaced by parquet metadata)",
                    V21_CONVERSION_HINT,
                )

    def validate_dataset_readme(self):
        """Validate the Open-H-Endoluminal required meta/README.md"""
        category = "Dataset README"
        readme_path = self.dataset_path / "meta" / "README.md"

        if not readme_path.exists():
            self.add_result(
                ValidationLevel.ERROR,
                category,
                "README.md not found in meta/ directory "
                "(Open-H-Endoluminal requirement)",
                "Complete templates/dataset_template.md and place it at "
                "meta/README.md, documenting the synchronization method and "
                "sample rates",
            )
            return

        self.add_result(
            ValidationLevel.SUCCESS,
            category,
            "README.md found in meta/ directory",
        )

        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            self.add_result(
                ValidationLevel.ERROR, category, f"Could not read meta/README.md: {e}"
            )
            return

        content_lower = content.lower()

        # Synchronization documentation is mandatory
        if "synchron" in content_lower or "timestamp" in content_lower:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                "Synchronization documentation found in meta/README.md",
            )
        else:
            self.add_result(
                ValidationLevel.ERROR,
                category,
                "meta/README.md does not document synchronization",
                "Every submission must document its synchronization method and "
                "sample rates (see templates/dataset_template.md)",
            )

        # Signal tier should be stated
        has_tier = bool(self.SIGNAL_TIER_REGEX.search(content)) or any(
            keyword in content_lower for keyword in self.SIGNAL_TIER_KEYWORDS
        )
        if has_tier:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                "Signal tier documented in meta/README.md",
            )
        else:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                "No signal tier mentioned in meta/README.md",
                "State the signal tier of the submission: S1 (native robot "
                "kinematics), S2 (tracked pose), S3 (inferred pose), or S4 "
                "(video with rich labels). Signal tiers weight contributed hours",
            )

    # ------------------------------------------------------------------
    # info.json checks
    # ------------------------------------------------------------------

    def validate_info_json(self):
        """Validate info.json content and Open-H-Endoluminal requirements"""
        category = "Dataset Info"
        info_path = self.dataset_path / "meta" / "info.json"

        if not info_path.exists():
            return

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except json.JSONDecodeError as e:
            self.add_result(
                ValidationLevel.ERROR, category, f"Invalid JSON in info.json: {e}"
            )
            return

        # Check codebase_version (dataset format version, not package version)
        codebase_version = info.get("codebase_version")
        if codebase_version is None:
            self.add_result(
                ValidationLevel.ERROR,
                category,
                "info.json is missing 'codebase_version' "
                "(expected a v3.x value such as \"v3.0\")",
            )
        elif not str(codebase_version).startswith("v3"):
            self.add_result(
                ValidationLevel.ERROR,
                category,
                f"codebase_version is '{codebase_version}' but Open-H-Endoluminal "
                "standardizes on LeRobot dataset format v3.0",
                V21_CONVERSION_HINT,
            )
        else:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"codebase_version '{codebase_version}' matches dataset format v3.0",
            )

        # Check FPS
        if "fps" in info:
            fps = info["fps"]
            if isinstance(fps, (int, float)) and fps < self.MIN_FPS:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"FPS ({fps}) below recommended minimum ({self.MIN_FPS} Hz)",
                    "Consider collecting data at ≥20 Hz for better quality",
                )
            elif isinstance(fps, (int, float)):
                self.add_result(
                    ValidationLevel.SUCCESS, category, f"FPS ({fps}) meets requirements"
                )
        else:
            self.add_result(
                ValidationLevel.WARNING, category, "No 'fps' value in info.json"
            )

        # Check robot type
        if "robot_type" not in info:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                "Robot type not specified in info.json",
                "State the device or platform (e.g. flexible endoscope, "
                "bronchoscope, catheter system)",
            )

        # Check splits (tolerated if absent in v3.0)
        if "splits" in info and isinstance(info["splits"], dict) and info["splits"]:
            self._validate_splits(info["splits"])
        else:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "No data splits found in info.json (v3.0 datasets may record "
                "splits elsewhere or omit them)",
                "If you define splits, use train/val/test plus optional "
                "'recovery' and 'failure' splits as episode-index ranges",
            )

        # Check features
        if "features" in info and isinstance(info["features"], dict):
            self._validate_features(info["features"])
        else:
            self.add_result(
                ValidationLevel.ERROR,
                category,
                "No 'features' mapping found in info.json",
            )

    def _validate_splits(self, splits: Dict):
        """Validate dataset splits including recovery/failure"""
        category = "Data Splits"

        # Check standard splits
        standard_splits = ["train", "val", "test"]
        for split in standard_splits:
            if split not in splits:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"Standard split '{split}' not defined",
                )

        # Check for recovery/failure splits (Open-H best practice)
        if "recovery" in splits:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                "Recovery examples split defined (Open-H best practice)",
            )
        else:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "Consider adding recovery examples for improved robustness",
            )

        if "failure" in splits:
            self.add_result(
                ValidationLevel.INFO, category, "Failure examples split defined"
            )

    def _validate_features(self, features: Dict):
        """Validate dataset features for Open-H-Endoluminal compliance"""
        category = "Dataset Features"

        has_action = "action" in features
        has_state = "observation.state" in features
        pose_meta_features = [
            k
            for k in features
            if k.startswith("observation.meta.") and "pose" in k.lower()
        ]
        rich_label_markers = (
            "segmentation",
            "mask",
            "depth",
            "phase",
            "label",
            "annotation",
            "vqa",
            "caption",
            "bbox",
        )
        rich_label_features = [
            k
            for k in features
            if k == "instruction.text"
            or any(marker in k.lower() for marker in rich_label_markers)
        ]

        if has_action and has_state:
            for required in self.REQUIRED_FEATURES:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"Required feature '{required}' present",
                )
        elif not has_action and not has_state:
            if pose_meta_features:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    "'action' and 'observation.state' are both missing, but a pose "
                    f"stream is present ({', '.join(pose_meta_features[:3])})",
                    "Tracked or inferred pose (signal tiers S2/S3) satisfies the "
                    "minimum signal requirement; add native kinematics as 'action' "
                    "and 'observation.state' if the platform provides them",
                )
            elif rich_label_features:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    "No action, state, or pose signal found, but rich-label "
                    f"features are present ({', '.join(rich_label_features[:3])})",
                    "This looks like a rich-annotations submission (signal tier "
                    "S4, weight x2 per the RFP). Document the S4 tier and the "
                    "annotation types in meta/README.md; raw video without such "
                    "annotations is not accepted",
                )
            else:
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    "'action' and 'observation.state' are both missing, and no "
                    "observation.meta.*pose* or rich-label feature exists",
                    "Raw, unlabeled video is not accepted by Open-H-Endoluminal. "
                    "Every submission needs a synchronized state or action signal "
                    "(native kinematics as 'action' and 'observation.state', or a "
                    "tracked or inferred pose stream such as "
                    "'observation.meta.em_pose'), or rich labels that qualify for "
                    "the S4 tier (segmentation, depth, phase, VQA, or similar)",
                )
        else:
            missing = "action" if not has_action else "observation.state"
            present = "observation.state" if not has_action else "action"
            self.add_result(
                ValidationLevel.WARNING,
                category,
                f"Required feature '{missing}' is missing ('{present}' is present)",
                "Provide both 'action' and 'observation.state' where the platform "
                "allows (insertion depth, shaft rotation, tip bend angles or tendon "
                "displacements, and tip pose for flexible robots)",
            )
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"Required feature '{present}' present",
            )

        # Check camera streams and their naming
        image_like_features = [
            k
            for k, v in features.items()
            if isinstance(v, dict) and v.get("dtype") in ("video", "image")
        ]
        misnamed_cameras = [
            k
            for k in image_like_features
            if not k.startswith(self.RECOMMENDED_IMAGE_PREFIX)
        ]
        if misnamed_cameras:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                "Camera stream(s) not using the recommended "
                f"'{self.RECOMMENDED_IMAGE_PREFIX}' prefix: "
                f"{', '.join(misnamed_cameras[:5])}",
                "Use 'observation.images.<view>' naming, e.g. "
                "observation.images.endoscope or observation.images.fluoro",
            )

        prefixed_image_features = [
            k for k in features if k.startswith(self.RECOMMENDED_IMAGE_PREFIX)
        ]
        if not prefixed_image_features:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                "No image features found with recommended prefix "
                f"'{self.RECOMMENDED_IMAGE_PREFIX}'",
                "Use 'observation.images.<view>' naming for camera views",
            )
        else:
            for img_feature in prefixed_image_features:
                self._validate_image_feature(img_feature, features[img_feature])

        # Check for endoluminal per-frame metadata
        meta_features = [k for k in features if k.startswith("observation.meta.")]
        if meta_features:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"Per-frame metadata features found: {', '.join(meta_features[:3])}",
            )
        else:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "Consider adding per-frame metadata under 'observation.meta.<field>' "
                "(e.g. observation.meta.scope_type, observation.meta.em_pose)",
            )

        # Check for timestep-level language
        if "instruction.text" in features:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                "Timestep-level language feature 'instruction.text' present",
            )
        else:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "Consider adding timestep-level language in 'instruction.text' "
                "(e.g. time-aligned narration of the sub-task)",
            )

        # Best-effort camera-frame kinematics for RGB endoscopy submissions.
        # The chip-on-tip camera is the end effector, so camera-frame motion is
        # the endoluminal equivalent of the camera-frame end-effector pose used
        # by rigid-arm datasets. Every observation.images.* key counts as an
        # RGB camera stream unless its name marks it as fluoroscopy, so generic
        # keys (observation.images.rgb, observation.images.cam0,
        # observation.images.main) also trigger the check; fluoroscopy-only
        # submissions are exempt and simply have no qualifying streams.
        fluoroscopy_markers = ("fluoro", "xray", "x_ray", "x-ray")
        rgb_camera_streams = [
            k
            for k in features
            if k.startswith(self.RECOMMENDED_IMAGE_PREFIX)
            and not any(marker in k.lower() for marker in fluoroscopy_markers)
        ]
        has_camera_frame_kinematics = any(
            "camera_frame" in k.lower() and "pose" in k.lower() for k in features
        )
        if rgb_camera_streams:
            if has_camera_frame_kinematics:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    "Camera-frame kinematics present for the RGB camera stream(s)",
                )
            else:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    "RGB camera stream(s) present "
                    f"({', '.join(rgb_camera_streams[:3])}) but no camera-frame "
                    "kinematics feature found",
                    "Submissions with an RGB camera stream must make a best "
                    "effort to provide kinematics as camera-frame motion under "
                    "'observation.meta.camera_frame_delta_pose' (per-step relative "
                    "camera pose expressed in the previous frame's optical "
                    "coordinates); fluoroscopy-only submissions are exempt. If "
                    "infeasible, justify it in meta/README.md",
                )

    def _validate_image_feature(self, feature_name: str, feature_info: Dict):
        """Validate individual image feature specifications"""
        category = "Image Features"

        if not isinstance(feature_info, dict):
            return

        # Check if video format is used (encouraged)
        if feature_info.get("dtype") == "video":
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"Feature '{feature_name}' uses efficient video format",
            )
        else:
            self.add_result(
                ValidationLevel.INFO,
                category,
                f"Feature '{feature_name}' does not use the 'video' dtype",
                "MP4 video encoding is encouraged for camera streams",
            )

        # Check resolution if shape is provided
        if "shape" in feature_info:
            shape = feature_info["shape"]
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                height, width = shape[0], shape[1]
                if height < self.MIN_RESOLUTION[0] or width < self.MIN_RESOLUTION[1]:
                    self.add_result(
                        ValidationLevel.WARNING,
                        category,
                        f"Feature '{feature_name}' resolution ({height}x{width}) "
                        "below recommended minimum",
                        f"Consider using ≥{self.MIN_RESOLUTION[0]}p resolution",
                    )

    # ------------------------------------------------------------------
    # Data file checks (v3.0 aggregated parquet files)
    # ------------------------------------------------------------------

    def validate_data_files(self):
        """Validate v3.0 data parquet files and their essential columns"""
        category = "Data Files"

        parquet_files = self._data_parquet_files()
        if not parquet_files:
            # Missing files already reported in directory structure check
            return

        pd = self._get_pandas()
        if pd is None:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "pandas/pyarrow not installed, skipping parquet column validation",
                "Install with: pip install pandas pyarrow",
            )
            return

        sample_file = parquet_files[0]
        try:
            df = pd.read_parquet(sample_file, engine="pyarrow")
        except Exception as e:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                f"Could not read data parquet file '{sample_file.name}': {str(e)[:100]}",
            )
            return

        # Essential columns for v3.0 data files, which aggregate multiple
        # episodes per parquet (episode_index is what separates them).
        essential_cols = [
            "episode_index",
            "frame_index",
            "timestamp",
            "index",
            "task_index",
        ]
        missing_cols = [col for col in essential_cols if col not in df.columns]
        for col in missing_cols:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                f"Data parquet missing expected column: {col}",
            )
        if not missing_cols:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                "All essential columns present in data parquet "
                f"({', '.join(essential_cols)})",
            )

        num_episodes_in_sample = None
        if "episode_index" in df.columns:
            try:
                num_episodes_in_sample = int(df["episode_index"].nunique())
            except Exception:
                num_episodes_in_sample = None

        message = (
            f"Sample data file '{sample_file.name}' readable with {len(df)} rows"
        )
        if num_episodes_in_sample is not None:
            message += f" across {num_episodes_in_sample} episode(s)"
        self.add_result(ValidationLevel.SUCCESS, category, message)

    # ------------------------------------------------------------------
    # Video checks
    # ------------------------------------------------------------------

    def validate_video_files(self):
        """Validate video files in the dataset"""
        category = "Video Files"
        videos_dir = self.dataset_path / "videos"

        if not videos_dir.exists():
            return

        video_files = list(videos_dir.glob("**/*.mp4"))

        if not video_files:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                "No MP4 video files found",
                "Ensure videos are properly encoded",
            )
            return

        self.add_result(
            ValidationLevel.SUCCESS, category, f"Found {len(video_files)} video files"
        )

        if cv2 is None:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "OpenCV (cv2) not installed, skipping per-video quality checks",
                "Install with: pip install opencv-python",
            )
            return

        # Sample check on first few videos
        sample_size = min(3, len(video_files))
        for video_path in video_files[:sample_size]:
            self._validate_video_file(video_path)

    def _validate_video_file(self, video_path: Path):
        """Validate individual video file"""
        category = "Video Quality"

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Cannot open video file: {video_path.name}",
                )
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.release()

            # Validate video properties
            if fps < self.MIN_FPS:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"Video '{video_path.name}' FPS ({fps:.1f}) below recommended minimum",
                )

            if height < self.MIN_RESOLUTION[0]:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"Video '{video_path.name}' resolution ({height}x{width}) below recommended",
                )

            if frame_count == 0:
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Video '{video_path.name}' has no frames",
                )

        except Exception as e:
            self.add_result(
                ValidationLevel.ERROR,
                category,
                f"Error validating video '{video_path.name}': {e}",
            )

    # ------------------------------------------------------------------
    # Episode metadata and task checks (v3.0 parquet metadata)
    # ------------------------------------------------------------------

    def validate_episodes(self):
        """Validate v3.0 episode metadata (meta/episodes/) and tasks.parquet"""
        category = "Episodes"

        episode_files = self._episode_metadata_files()
        if not episode_files:
            # Missing directory or files already reported in metadata check
            return

        pd = self._get_pandas()
        if pd is None:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "pandas/pyarrow not installed, skipping episode metadata validation",
                "Install with: pip install pandas pyarrow",
            )
            return

        total_episodes = 0
        for ep_file in episode_files:
            try:
                ep_df = pd.read_parquet(ep_file, engine="pyarrow")
                total_episodes += len(ep_df)
            except Exception as e:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"Could not read episode metadata file '{ep_file.name}': "
                    f"{str(e)[:100]}",
                )

        if total_episodes == 0:
            self.add_result(
                ValidationLevel.ERROR,
                category,
                "No episodes found in meta/episodes/ metadata",
            )
            return

        self.add_result(
            ValidationLevel.SUCCESS,
            category,
            f"Found {total_episodes} episode(s) across {len(episode_files)} "
            "metadata file(s)",
        )

        # Check task descriptions in meta/tasks.parquet
        tasks_path = self.dataset_path / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            return

        try:
            tasks_df = pd.read_parquet(tasks_path, engine="pyarrow")
        except Exception as e:
            self.add_result(
                ValidationLevel.ERROR,
                category,
                f"Error reading meta/tasks.parquet: {str(e)[:100]}",
            )
            return

        tasks = set()
        if "task" in tasks_df.columns:
            tasks.update(str(t) for t in tasks_df["task"].dropna().tolist())
        else:
            # Some v3.0 datasets index tasks.parquet by the task string itself
            tasks.update(str(t) for t in tasks_df.index.tolist())

        if not tasks:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                "No task descriptions found in meta/tasks.parquet",
                "Every submission must state task intent and the target "
                "(navigation, screening/coverage, detection/diagnosis, or "
                "intervention); navigation data must show movement toward a "
                "stated target",
            )
        else:
            preview = ", ".join(sorted(tasks)[:5])
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"Found {len(tasks)} unique task(s): {preview}",
            )

            # Check for recovery tasks (Open-H best practice)
            recovery_tasks = [task for task in tasks if "recovery" in task.lower()]
            if recovery_tasks:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"Found {len(recovery_tasks)} recovery task(s), "
                    "excellent for robustness",
                )

    # ------------------------------------------------------------------
    # Hours accounting (hours are the contribution unit)
    # ------------------------------------------------------------------

    def validate_hours_accounting(self):
        """Compute hours of synchronized data, the contribution unit"""
        category = "Hours Accounting"

        info = self._load_info()
        fps = None
        total_frames = None
        if info:
            raw_fps = info.get("fps")
            if isinstance(raw_fps, (int, float)) and raw_fps > 0:
                fps = float(raw_fps)
            raw_total = info.get("total_frames")
            if isinstance(raw_total, (int, float)) and raw_total > 0:
                total_frames = float(raw_total)

        if fps and total_frames:
            hours = total_frames / fps / 3600.0
            self.report.total_hours = hours
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"info.json accounting: {int(total_frames)} frames at {fps:g} Hz "
                f"= {hours:.2f} hours of synchronized data",
                "Hours are the Open-H-Endoluminal contribution unit; per-setting "
                "minimums are listed in the RFP "
                "(assets/open-h-endoluminal-rfp.pdf, Section 3.6)",
            )
        else:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                "Cannot compute total hours from info.json "
                "(requires positive 'total_frames' and 'fps' values)",
                "Hours of synchronized data are the Open-H-Endoluminal "
                "contribution unit; make sure info.json reports total_frames and fps",
            )

        # Per-episode accounting from meta/episodes metadata where available
        pd = self._get_pandas()
        episode_files = self._episode_metadata_files()
        if pd is None or not episode_files or not fps:
            return

        total_length = 0
        length_found = False
        for ep_file in episode_files:
            try:
                ep_df = pd.read_parquet(ep_file, engine="pyarrow")
            except Exception:
                continue
            if "length" in ep_df.columns:
                length_found = True
                try:
                    total_length += float(ep_df["length"].sum())
                except Exception:
                    pass

        if length_found and total_length > 0:
            episode_hours = total_length / fps / 3600.0
            self.report.episode_hours = episode_hours
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"Per-episode accounting: {int(total_length)} frames summed over "
                f"episode lengths = {episode_hours:.2f} hours",
            )
            if self.report.total_hours is not None:
                relative_gap = abs(episode_hours - self.report.total_hours) / max(
                    self.report.total_hours, 1e-9
                )
                if relative_gap > 0.01:
                    self.add_result(
                        ValidationLevel.WARNING,
                        category,
                        "Hours mismatch: info.json accounting "
                        f"({self.report.total_hours:.2f} h) differs from the "
                        f"per-episode sum ({episode_hours:.2f} h) by "
                        f"{relative_gap:.1%}",
                        "Check that info.json total_frames matches the episode "
                        "metadata",
                    )
        else:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "No per-episode 'length' column found in meta/episodes/ metadata; "
                "skipping per-episode hours accounting",
            )

    # ------------------------------------------------------------------
    # Timestamp integrity suite
    # ------------------------------------------------------------------

    def validate_timestamps(self):
        """Validate timestamp column in v3.0 data parquet files.

        Checks for issues known to cause training/inference failures in
        downstream models:
        - Absolute Unix epoch timestamps stored as float32 (precision collapse)
        - Constant or near-constant timestamps across an episode
        - Non-monotonic timestamps
        - Non-strictly-monotonic timestamps (duplicate values)
        - Unreasonable spacing relative to declared FPS
        - Timestamps not relative to episode start

        Format v3.0 aggregates multiple episodes per parquet file, so each
        file is grouped by episode_index and every episode is checked
        independently.
        """
        category = "Timestamps"
        data_dir = self.dataset_path / "data"

        if not data_dir.exists():
            self.add_result(
                ValidationLevel.ERROR,
                category,
                "Data directory not found, cannot validate timestamps",
            )
            return

        pd = self._get_pandas()
        if pd is None:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "pandas not installed, skipping timestamp validation",
                "Install with: pip install pandas pyarrow",
            )
            return

        # Read FPS from info.json for spacing checks
        fps = None
        info_path = self.dataset_path / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                raw_fps = info.get("fps")
                if raw_fps is not None:
                    try:
                        fps = float(raw_fps)
                        if fps <= 0:
                            self.add_result(
                                ValidationLevel.WARNING,
                                category,
                                f"Invalid fps value in info.json: {raw_fps}. "
                                "Skipping FPS-dependent timestamp checks.",
                            )
                            fps = None
                    except (TypeError, ValueError):
                        self.add_result(
                            ValidationLevel.WARNING,
                            category,
                            f"Non-numeric fps value in info.json: {raw_fps}. "
                            "Skipping FPS-dependent timestamp checks.",
                        )
                        fps = None
            except (json.JSONDecodeError, KeyError):
                pass

        parquet_files = self._data_parquet_files()
        if not parquet_files:
            return

        files_checked = 0
        # episodes_attempted counts every episode (or file treated as one
        # episode) that was examined, including those that fail early;
        # episodes_checked counts only those that passed the preliminary
        # column/dtype/finiteness checks and ran the full timestamp suite.
        episodes_attempted = 0
        episodes_checked = 0
        episodes_with_errors = 0
        episodes_with_warnings = 0
        issue_summary = {
            "missing_timestamp_column": [],
            "non_numeric_dtype": [],
            "non_finite_values": [],
            "epoch_timestamps": [],
            "constant_timestamps": [],
            "low_uniqueness": [],
            "non_monotonic": [],
            "non_strictly_monotonic": [],
            "bad_spacing": [],
            "not_relative": [],
        }

        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf, engine="pyarrow")
            except Exception as e:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"Could not read {pf.name}: {e}",
                )
                continue

            files_checked += 1

            if "timestamp" not in df.columns:
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"{pf.name}: missing 'timestamp' column",
                )
                issue_summary["missing_timestamp_column"].append(pf.name)
                # Without a timestamp column the file's episodes cannot be
                # checked individually, so the whole file counts as a single
                # examined unit.
                episodes_attempted += 1
                episodes_with_errors += 1
                continue

            # v3.0 data files aggregate multiple episodes per parquet;
            # group by episode_index so each episode is checked on its own.
            if "episode_index" in df.columns:
                episode_groups = [
                    (ep_idx, ep_df) for ep_idx, ep_df in df.groupby("episode_index")
                ]
            else:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"{pf.name}: missing 'episode_index' column, treating the whole "
                    "file as a single episode (v3.0 data files aggregate multiple "
                    "episodes per parquet)",
                )
                episode_groups = [(None, df)]

            for ep_idx, ep_df in episode_groups:
                if ep_idx is None:
                    ep_name = pf.name
                else:
                    ep_name = f"{pf.name}[episode {ep_idx}]"

                episodes_attempted += 1

                ts_series = ep_df["timestamp"]
                if not np.issubdtype(ts_series.dtype, np.number):
                    self.add_result(
                        ValidationLevel.ERROR,
                        category,
                        f"{ep_name}: timestamp column has non-numeric dtype "
                        f"({ts_series.dtype})",
                    )
                    issue_summary["non_numeric_dtype"].append(ep_name)
                    episodes_with_errors += 1
                    continue

                ts = pd.to_numeric(ts_series, errors="coerce").to_numpy(
                    dtype=np.float64
                )
                non_finite_count = int(np.sum(~np.isfinite(ts)))
                if non_finite_count > 0:
                    self.add_result(
                        ValidationLevel.ERROR,
                        category,
                        f"{ep_name}: timestamp contains {non_finite_count} "
                        "NaN/Inf value(s)",
                    )
                    issue_summary["non_finite_values"].append(ep_name)
                    episodes_with_errors += 1
                    continue

                episodes_checked += 1
                n = len(ts)
                has_error = False
                has_warning = False

                if n < 2:
                    continue

                # --- Check 1: Absolute Unix epoch timestamps ---
                # float32 can only represent ~7 significant digits; Unix epoch
                # values (~1.7e9) lose all sub-second precision, collapsing
                # per-frame deltas to zero.
                if ts[0] > 1e6:
                    issue_summary["epoch_timestamps"].append(ep_name)
                    is_float32 = ts_series.dtype == np.float32
                    if is_float32:
                        self.add_result(
                            ValidationLevel.ERROR,
                            category,
                            f"{ep_name}: timestamps are absolute Unix epoch values "
                            f"(ts[0]={ts[0]:.1f}) stored as float32; precision "
                            "collapse makes per-frame deltas invisible to "
                            "downstream models",
                            "Rewrite the dataset through the LeRobot API (the "
                            "canonical timestamp column is frame_index / fps) "
                            "and preserve epoch clocks as the "
                            "observation.meta.host_stamp_ns feature (int64 ns)",
                        )
                        has_error = True
                    else:
                        self.add_result(
                            ValidationLevel.WARNING,
                            category,
                            f"{ep_name}: timestamps appear to be absolute Unix "
                            f"epoch values (ts[0]={ts[0]:.1f}); downstream models "
                            "expect relative timestamps starting near 0",
                            "The canonical timestamp column should be the "
                            "frame timeline (frame_index / fps); keep epoch "
                            "clocks in observation.meta.host_stamp_ns instead",
                        )
                        has_warning = True
                        issue_summary["not_relative"].append(ep_name)

                # --- Check 2: Constant or near-constant timestamps ---
                ts_min = float(np.min(ts))
                ts_max = float(np.max(ts))
                ts_range = ts_max - ts_min
                expected_duration = (n - 1) / fps if fps else None

                if ts_range == 0:
                    issue_summary["constant_timestamps"].append(ep_name)
                    self.add_result(
                        ValidationLevel.ERROR,
                        category,
                        f"{ep_name}: all {n} timestamps are identical "
                        f"({ts[0]:.6f}); video frame selection will always "
                        "return frame 0",
                    )
                    has_error = True
                elif expected_duration and ts_range < expected_duration * 0.01:
                    issue_summary["constant_timestamps"].append(ep_name)
                    self.add_result(
                        ValidationLevel.ERROR,
                        category,
                        f"{ep_name}: timestamp range ({ts_range:.2e}s) is "
                        "negligible compared to expected episode duration "
                        f"({expected_duration:.2f}s at {fps} fps), "
                        "effectively constant",
                    )
                    has_error = True

                # --- Check 3: Uniqueness ---
                num_unique = len(np.unique(ts))
                uniqueness_ratio = num_unique / n

                if num_unique == 1 and n > 1:
                    pass  # already reported in constant check
                elif uniqueness_ratio < 0.5:
                    issue_summary["low_uniqueness"].append(ep_name)
                    self.add_result(
                        ValidationLevel.ERROR,
                        category,
                        f"{ep_name}: only {num_unique}/{n} unique timestamp values "
                        f"({uniqueness_ratio:.1%}); most frames share timestamps, "
                        "causing incorrect video frame lookups",
                    )
                    has_error = True
                elif uniqueness_ratio < 1.0:
                    issue_summary["non_strictly_monotonic"].append(ep_name)
                    num_duplicates = n - num_unique
                    self.add_result(
                        ValidationLevel.WARNING,
                        category,
                        f"{ep_name}: {num_duplicates} duplicate timestamp value(s) "
                        f"({num_unique}/{n} unique, {uniqueness_ratio:.1%}); "
                        "ideally each frame should have a distinct timestamp",
                    )
                    has_warning = True

                # --- Check 4: Monotonicity ---
                diffs = np.diff(ts)
                num_decreasing = int(np.sum(diffs < 0))
                if num_decreasing > 0:
                    issue_summary["non_monotonic"].append(ep_name)
                    first_decrease_idx = int(np.argmax(diffs < 0))
                    self.add_result(
                        ValidationLevel.ERROR,
                        category,
                        f"{ep_name}: timestamps are NOT monotonically increasing; "
                        f"{num_decreasing} decrease(s) found "
                        f"(first at index {first_decrease_idx}: "
                        f"{ts[first_decrease_idx]:.6f} -> {ts[first_decrease_idx+1]:.6f})",
                    )
                    has_error = True

                # --- Check 5: Spacing relative to FPS ---
                if fps and ts_range > 0 and num_unique > 1:
                    expected_spacing = 1.0 / fps
                    positive_diffs = diffs[diffs > 0]
                    if len(positive_diffs) > 0:
                        mean_spacing = float(np.mean(positive_diffs))
                        ratio = mean_spacing / expected_spacing
                        if ratio > 5.0 or ratio < 0.1:
                            issue_summary["bad_spacing"].append(ep_name)
                            self.add_result(
                                ValidationLevel.WARNING,
                                category,
                                f"{ep_name}: mean timestamp spacing "
                                f"({mean_spacing:.4f}s) deviates significantly "
                                f"from expected 1/{fps}={expected_spacing:.4f}s "
                                f"(ratio: {ratio:.1f}x)",
                                "This may indicate timestamps in wrong units or "
                                "from a different clock source",
                            )
                            has_warning = True

                # --- Check 6: Relative timestamps (should start near 0) ---
                if 0 < ts[0] <= 1e6:
                    if ts[0] > 60.0:
                        issue_summary["not_relative"].append(ep_name)
                        self.add_result(
                            ValidationLevel.WARNING,
                            category,
                            f"{ep_name}: first timestamp is {ts[0]:.2f}s; "
                            "timestamps may not be relative to episode start",
                            "LeRobot writes the canonical timestamp column as "
                            "frame_index/fps (starting at 0.0); raw capture "
                            "clocks belong in observation.meta.host_stamp_ns",
                        )
                        has_warning = True

                if has_error:
                    episodes_with_errors += 1
                elif has_warning:
                    episodes_with_warnings += 1

        # --- Aggregate summary ---
        if episodes_attempted == 0:
            # Nothing was examined (e.g. no readable parquet files); any
            # read failures were already reported above.
            return

        total_issues = episodes_with_errors + episodes_with_warnings
        if total_issues == 0:
            if episodes_checked > 0:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"All {episodes_checked} checked episodes have valid timestamps "
                    f"(monotonically increasing, unique per frame, relative to episode start)",
                )
        else:
            if episodes_with_errors > 0:
                broken_types = []
                if issue_summary["missing_timestamp_column"]:
                    broken_types.append(
                        f"{len(issue_summary['missing_timestamp_column'])} file(s) missing the timestamp column"
                    )
                if issue_summary["non_numeric_dtype"]:
                    broken_types.append(
                        f"{len(issue_summary['non_numeric_dtype'])} with a non-numeric timestamp dtype"
                    )
                if issue_summary["non_finite_values"]:
                    broken_types.append(
                        f"{len(issue_summary['non_finite_values'])} with NaN/Inf timestamp values"
                    )
                if issue_summary["epoch_timestamps"]:
                    broken_types.append(
                        f"{len(issue_summary['epoch_timestamps'])} with absolute epoch values"
                    )
                if issue_summary["constant_timestamps"]:
                    broken_types.append(
                        f"{len(issue_summary['constant_timestamps'])} with constant/collapsed timestamps"
                    )
                if issue_summary["low_uniqueness"]:
                    broken_types.append(
                        f"{len(issue_summary['low_uniqueness'])} with very low uniqueness"
                    )
                if issue_summary["non_monotonic"]:
                    broken_types.append(
                        f"{len(issue_summary['non_monotonic'])} with non-monotonic values"
                    )
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Timestamp issues found in {episodes_with_errors}/{episodes_attempted} "
                    f"examined episodes: {'; '.join(broken_types)}",
                    "Broken timestamps may cause downstream models "
                    "to always select the same video frame, "
                    "producing static/frozen training data. Fix the dataset's "
                    "timestamp column or ensure the training pipeline has a "
                    "fallback (e.g. frame_index / fps).",
                )

            if episodes_with_warnings > 0:
                warn_types = []
                if issue_summary["non_strictly_monotonic"]:
                    warn_types.append(
                        f"{len(issue_summary['non_strictly_monotonic'])} with duplicate timestamps"
                    )
                if issue_summary["bad_spacing"]:
                    warn_types.append(
                        f"{len(issue_summary['bad_spacing'])} with unexpected spacing"
                    )
                if issue_summary["not_relative"]:
                    warn_types.append(
                        f"{len(issue_summary['not_relative'])} with non-relative timestamps"
                    )
                if warn_types:
                    self.add_result(
                        ValidationLevel.WARNING,
                        category,
                        f"Timestamp warnings in {episodes_with_warnings}/{episodes_attempted} "
                        f"examined episodes: {'; '.join(warn_types)}",
                    )

        self.add_result(
            ValidationLevel.INFO,
            category,
            f"Examined {episodes_attempted} episode(s) across {files_checked} "
            f"data parquet file(s) for timestamp integrity "
            f"({episodes_checked} passed the preliminary timestamp checks).",
        )

    # ------------------------------------------------------------------
    # Synchronization checks
    # ------------------------------------------------------------------

    def validate_data_synchronization(self):
        """Check for data synchronization parameters in info.json"""
        category = "Data Synchronization"

        info = self._load_info()
        if info is None:
            return

        # Check for tolerance_s parameter (indicates sync consideration)
        if "tolerance_s" in info:
            tolerance = info["tolerance_s"]
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"Synchronization tolerance specified: {tolerance}s",
            )
        else:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "No 'tolerance_s' value in info.json",
                "Record the synchronization tolerance via tolerance_s "
                "(typical 0.1 s)",
            )

    # ------------------------------------------------------------------
    # Final load check (the only check that uses the lerobot package)
    # ------------------------------------------------------------------

    def validate_lerobot_compatibility(self):
        """Validate the dataset loads with the lerobot package locally.

        This is the only check that requires lerobot. When the package is
        absent the check degrades to a WARNING; every other check in this
        script has already run without it.
        """
        category = "LeRobot Compatibility"

        try:
            installed_version = get_version("lerobot")
        except PackageNotFoundError:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                "lerobot is not installed; skipping the dataset load check "
                "(all structural, metadata, timestamp, and video checks above "
                "still ran)",
                "Install with: pip install 'lerobot[dataset]==0.6.0' (the "
                "version pinned by the contribution guide; Python >= 3.12). "
                "Note that the lerobot PACKAGE version and the dataset FORMAT "
                "version are separate schemes: any package >= 0.4.0 reads "
                f"dataset format v3.0. {V21_CONVERSION_HINT}",
            )
            return

        if _parse_version(installed_version) < _parse_version(MIN_LEROBOT_VERSION):
            self.add_result(
                ValidationLevel.ERROR,
                category,
                f"lerobot {installed_version} is too old for dataset format v3.0; "
                f"lerobot>={MIN_LEROBOT_VERSION} is required",
                "Upgrade with: pip install 'lerobot[dataset]==0.6.0' (the "
                "version pinned by the contribution guide). The lerobot PACKAGE "
                "version and the dataset FORMAT version are separate versioning "
                f"schemes: any package >= 0.4.0 reads dataset format v3.0. "
                f"{V21_CONVERSION_HINT}",
            )
            return

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception as e:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                f"Could not import LeRobotDataset from lerobot {installed_version}: "
                f"{e}. Skipping the dataset load check",
                "Reinstall with: pip install 'lerobot[dataset]==0.6.0' (on "
                "lerobot >= 0.5 the [dataset] extra is required for dataset "
                "support)",
            )
            return

        try:
            # Load dataset from local path only, no remote access.
            # Create repo_id in the form: parent_folder/child_folder
            repo_id = "/".join(str(self.dataset_path).split("/")[-2:])

            dataset = LeRobotDataset(
                repo_id,
                root=str(self.dataset_path),
            )

            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                "Dataset structure is compatible with LeRobot dataset format v3.0",
            )

            # Check dataset properties
            if hasattr(dataset, "__len__"):
                dataset_len = len(dataset)
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"Dataset contains {dataset_len} frames",
                )

            # Check if dataset has required attributes
            if hasattr(dataset, "features"):
                self.add_result(
                    ValidationLevel.SUCCESS, category, "Dataset features accessible"
                )

        except Exception as e:
            self.add_result(
                ValidationLevel.ERROR,
                category,
                f"Error loading dataset with lerobot {installed_version}: {e}",
                V21_CONVERSION_HINT,
            )

    # ------------------------------------------------------------------
    # Orchestration and reporting
    # ------------------------------------------------------------------

    def run_validation(self) -> ValidationReport:
        """Run all validation checks"""
        print(f"\n{'='*60}")
        print("Open-H-Endoluminal Dataset Validation")
        print("Target format: LeRobot dataset format v3.0")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"{'='*60}\n")

        # Run all validation checks
        print("🔍 Validating directory structure...")
        self.validate_directory_structure()

        print("📁 Validating metadata files...")
        self.validate_metadata_files()

        print("📖 Validating dataset README...")
        self.validate_dataset_readme()

        print("📊 Validating dataset info...")
        self.validate_info_json()

        print("🗂️ Validating data files...")
        self.validate_data_files()

        print("🎬 Validating video files...")
        self.validate_video_files()

        print("📝 Validating episodes...")
        self.validate_episodes()

        print("⏳ Computing hours of synchronized data...")
        self.validate_hours_accounting()

        print("🕐 Validating timestamps...")
        self.validate_timestamps()

        print("⏱️ Validating synchronization...")
        self.validate_data_synchronization()

        print("🤖 Validating LeRobot compatibility...")
        self.validate_lerobot_compatibility()

        return self.report

    def print_summary(self):
        """Print validation summary"""
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")

        # Hours are the Open-H-Endoluminal contribution unit: show prominently.
        print("\n⏳ Hours of synchronized data (the contribution unit):")
        if self.report.total_hours is not None:
            print(
                f"  Total (info.json, total_frames / fps): "
                f"{self.report.total_hours:.2f} hours"
            )
        if self.report.episode_hours is not None:
            print(
                f"  Total (per-episode length sum / fps):  "
                f"{self.report.episode_hours:.2f} hours"
            )
        if self.report.total_hours is None and self.report.episode_hours is None:
            print("  Could not be computed (info.json needs total_frames and fps)")
        print(
            "  Per-setting hour minimums are listed in the RFP "
            "(assets/open-h-endoluminal-rfp.pdf, Section 3.6)."
        )

        # Count results by level
        print("\n📊 Results Overview:")
        print(f"  ✅ Success: {self.report.success_count}")
        print(f"  ℹ️  Info: {self.report.info_count}")
        print(f"  ⚠️  Warnings: {self.report.warning_count}")
        print(f"  ❌ Errors: {self.report.error_count}")

        # Group results by category
        categories = {}
        for result in self.report.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        # Print errors and warnings by category
        if self.report.error_count > 0:
            print("\n🚨 Critical Issues (Must Fix):")
            for category, results in categories.items():
                errors = [r for r in results if r.level == ValidationLevel.ERROR]
                if errors:
                    print(f"\n  {category}:")
                    for error in errors:
                        print(f"    • {error.message}")
                        if error.details:
                            print(f"      → {error.details}")

        if self.report.warning_count > 0:
            print("\n⚠️  Recommendations (Should Fix):")
            for category, results in categories.items():
                warning_results = [
                    r for r in results if r.level == ValidationLevel.WARNING
                ]
                if warning_results:
                    print(f"\n  {category}:")
                    for warning in warning_results:
                        print(f"    • {warning.message}")
                        if warning.details:
                            print(f"      → {warning.details}")

        # Final verdict
        print(f"\n{'='*60}")
        print("FINAL VERDICT")
        print(f"{'='*60}")

        if self.report.is_compliant:
            print("\n✅ Dataset is Open-H-Endoluminal READY!")
            print(
                "   The dataset meets all requirements for the Open-H-Endoluminal "
                "data collection initiative."
            )
            if self.report.warning_count > 0:
                print(
                    f"   Consider addressing {self.report.warning_count} warning(s) for best practices."
                )
        else:
            print("\n❌ Dataset is NOT Open-H-Endoluminal ready.")
            print(
                f"   {self.report.error_count} critical issue(s) must be fixed before the dataset"
            )
            print("   can be considered compliant with Open-H-Endoluminal requirements.")
            print(
                "\n   Please address the errors listed above and run validation again."
            )

        print(f"\n{'='*60}\n")


def main():
    """Main entry point for the validation script"""
    parser = argparse.ArgumentParser(
        description=(
            "Validate LeRobot datasets (format v3.0) for Open-H-Endoluminal "
            "compliance (LOCAL validation only)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool performs LOCAL validation only and does not require internet access.
Run this as a final check before uploading your dataset.

Structural, metadata, timestamp, and video checks require only
pyarrow/pandas, numpy, and OpenCV. The final load check uses the lerobot
package (>=0.4.0) and is skipped with a warning if it is not installed.

Examples:
  # Validate local dataset directory
  python scripts/validation/validate_formatting.py /path/to/dataset

  # Enable verbose output for detailed results
  python scripts/validation/validate_formatting.py /path/to/dataset --verbose
        """,
    )

    # Dataset path argument
    parser.add_argument(
        "dataset_path", type=Path, help="Path to the local LeRobot dataset directory"
    )

    # Options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output showing all validation results",
    )

    args = parser.parse_args()

    try:
        # Create and run validator
        validator = OpenHEndoluminalValidator(
            dataset_path=args.dataset_path, verbose=args.verbose
        )

        # Run validation
        report = validator.run_validation()

        # Print summary
        validator.print_summary()

        # Exit with appropriate code
        sys.exit(0 if report.is_compliant else 1)

    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
