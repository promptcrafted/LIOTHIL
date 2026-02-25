"""Dimljus dataset validation and organization.

Standalone tools for validating complete training datasets: pairing targets
with signals, checking quality, previewing bucketing, generating manifests,
and organizing into trainer-ready directories.
Works with any trainer — musubi-tuner, ai-toolkit, or Dimljus itself.

Quick start:
    python -m dimljus.dataset validate ./my_dataset
    python -m dimljus.dataset organize ./my_dataset -o ./clean -t musubi
"""

from dimljus.dataset.bucketing import (
    BucketAssignment,
    BucketGroup,
    BucketingResult,
    compute_bucket_key,
    preview_bucketing,
)
from dimljus.dataset.discover import (
    detect_structure,
    discover_all_datasets,
    discover_dataset,
    discover_files,
    pair_samples,
)
from dimljus.dataset.errors import (
    DatasetValidationError,
    DimljusDatasetError,
    OrganizeError,
)
from dimljus.dataset.manifest import (
    build_manifest,
    read_manifest,
    write_manifest,
)
from dimljus.dataset.models import (
    DatasetReport,
    DatasetValidation,
    OrganizeLayout,
    OrganizeResult,
    OrganizedSample,
    SamplePair,
    StructureType,
)
from dimljus.dataset.organize import (
    organize_dataset,
)
from dimljus.dataset.validate import (
    validate_all,
    validate_dataset,
    validate_sample,
)

__all__ = [
    # Errors
    "DatasetValidationError",
    "DimljusDatasetError",
    "OrganizeError",
    # Models
    "DatasetReport",
    "DatasetValidation",
    "OrganizeLayout",
    "OrganizeResult",
    "OrganizedSample",
    "SamplePair",
    "StructureType",
    # Bucketing
    "BucketAssignment",
    "BucketGroup",
    "BucketingResult",
    "compute_bucket_key",
    "preview_bucketing",
    # Discovery
    "detect_structure",
    "discover_all_datasets",
    "discover_dataset",
    "discover_files",
    "pair_samples",
    # Manifest
    "build_manifest",
    "read_manifest",
    "write_manifest",
    # Organize
    "organize_dataset",
    # Validation
    "validate_all",
    "validate_dataset",
    "validate_sample",
]
