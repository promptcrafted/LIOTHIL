"""Tests for dimljus.training.errors — error hierarchy."""

import pytest

from dimljus.training.errors import (
    CheckpointError,
    DimljusTrainingError,
    LoRAError,
    ModelBackendError,
    PhaseConfigError,
    ResumptionError,
    SamplingError,
)


class TestErrorHierarchy:
    """All training errors inherit from DimljusTrainingError."""

    def test_base_is_exception(self):
        assert issubclass(DimljusTrainingError, Exception)

    def test_phase_config_error_inherits(self):
        assert issubclass(PhaseConfigError, DimljusTrainingError)

    def test_checkpoint_error_inherits(self):
        assert issubclass(CheckpointError, DimljusTrainingError)

    def test_model_backend_error_inherits(self):
        assert issubclass(ModelBackendError, DimljusTrainingError)

    def test_lora_error_inherits(self):
        assert issubclass(LoRAError, DimljusTrainingError)

    def test_sampling_error_inherits(self):
        assert issubclass(SamplingError, DimljusTrainingError)

    def test_resumption_error_inherits(self):
        assert issubclass(ResumptionError, DimljusTrainingError)


class TestErrorMessages:
    """Error messages include the detail string."""

    def test_phase_config_error_message(self):
        err = PhaseConfigError("bad override chain")
        assert "Phase config error" in str(err)
        assert "bad override chain" in str(err)
        assert err.detail == "bad override chain"

    def test_checkpoint_error_message(self):
        err = CheckpointError("disk full")
        assert "Checkpoint error" in str(err)
        assert "disk full" in str(err)
        assert err.detail == "disk full"

    def test_model_backend_error_message(self):
        err = ModelBackendError("model not found")
        assert "Model backend error" in str(err)
        assert "model not found" in str(err)
        assert err.detail == "model not found"

    def test_lora_error_message(self):
        err = LoRAError("rank mismatch")
        assert "LoRA error" in str(err)
        assert "rank mismatch" in str(err)
        assert err.detail == "rank mismatch"

    def test_sampling_error_message(self):
        err = SamplingError("inference failed")
        assert "Sampling error" in str(err)
        assert "inference failed" in str(err)
        assert err.detail == "inference failed"

    def test_resumption_error_message(self):
        err = ResumptionError("corrupt state")
        assert "Resumption error" in str(err)
        assert "corrupt state" in str(err)
        assert err.detail == "corrupt state"


class TestErrorCatching:
    """Errors can be caught by base class."""

    def test_catch_phase_config_as_base(self):
        with pytest.raises(DimljusTrainingError):
            raise PhaseConfigError("test")

    def test_catch_checkpoint_as_base(self):
        with pytest.raises(DimljusTrainingError):
            raise CheckpointError("test")

    def test_catch_all_as_exception(self):
        with pytest.raises(Exception):
            raise LoRAError("test")
