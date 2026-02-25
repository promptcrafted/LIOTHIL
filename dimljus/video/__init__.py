"""Dimljus video ingestion and scene detection.

Standalone tools for going from raw footage to clean training clips.
Works with any trainer — musubi-tuner, ai-toolkit, or Dimljus itself.

Quick start:
    from dimljus.video import probe_video, validate_clip
    from dimljus.config import DimljusDataConfig

    meta = probe_video("clip.mp4")
    config = DimljusDataConfig(datasets=[{"path": "."}])
    result = validate_clip(meta, config.video)
"""

# Auto-discover ffmpeg if installed via WinGet/Chocolatey/Scoop
import dimljus.video._ffmpeg  # noqa: F401

from dimljus.video.errors import (
    DimljusVideoError,
    ExtractionError,
    FFmpegNotFoundError,
    ProbeError,
    SceneDetectNotFoundError,
    SplitError,
)
from dimljus.video.extract_models import (
    ExtractionConfig,
    ExtractionReport,
    ExtractionResult,
    ExtractionStrategy,
    ImageValidation,
)
from dimljus.video.models import (
    ClipInfo,
    ClipValidation,
    IssueCode,
    ScanReport,
    SceneBoundary,
    Severity,
    ValidationIssue,
    VideoMetadata,
)
from dimljus.video.probe import probe_directory, probe_video
from dimljus.video.validate import (
    format_scan_report,
    nearest_valid_frame_count,
    validate_clip,
    validate_directory,
)

__all__ = [
    # Errors
    "DimljusVideoError",
    "ExtractionError",
    "FFmpegNotFoundError",
    "ProbeError",
    "SceneDetectNotFoundError",
    "SplitError",
    # Video models
    "ClipInfo",
    "ClipValidation",
    "IssueCode",
    "ScanReport",
    "SceneBoundary",
    "Severity",
    "ValidationIssue",
    "VideoMetadata",
    # Extraction models
    "ExtractionConfig",
    "ExtractionReport",
    "ExtractionResult",
    "ExtractionStrategy",
    "ImageValidation",
    # Probe
    "probe_video",
    "probe_directory",
    # Validate
    "validate_clip",
    "validate_directory",
    "nearest_valid_frame_count",
    "format_scan_report",
]
