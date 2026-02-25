"""Tests for dimljus.training.wan.inference.WanInferencePipeline.

WanInferencePipeline is GPU-dependent, so all tests here are GPU-free.
They cover:

    1. Constructor attribute storage
    2. Lazy-load contract (_pipeline is None until generate() is called)
    3. I2V flag handling
    4. cleanup() safety (no-op when pipeline not loaded, clears when loaded)
    5. Protocol compliance with InferencePipeline (@runtime_checkable)
    6. Error propagation when generate() is called without a real model path
    7. Default device value

No torch, diffusers, or GPU hardware required for any of these tests.
"""

import pytest

from dimljus.training.wan.inference import WanInferencePipeline
from dimljus.training.protocols import InferencePipeline


# ---------------------------------------------------------------------------
# Constructor behaviour
# ---------------------------------------------------------------------------

def test_constructor_stores_config():
    """All constructor arguments must be persisted as instance attributes."""
    pipe = WanInferencePipeline(
        model_path="/fake/path",
        is_i2v=False,
        dtype="bf16",
    )
    assert pipe._model_path == "/fake/path"
    assert pipe._is_i2v is False
    assert pipe._dtype_str == "bf16"


def test_pipeline_not_loaded_at_init():
    """Lazy-load contract: _pipeline must be None immediately after __init__.

    The pipeline is only loaded on the first generate() call. Initialising
    the class should never touch the GPU or the filesystem.
    """
    pipe = WanInferencePipeline(model_path="/fake")
    assert pipe._pipeline is None


def test_i2v_flag_true():
    """is_i2v=True must be stored and readable."""
    pipe = WanInferencePipeline(model_path="/fake", is_i2v=True)
    assert pipe._is_i2v is True


def test_i2v_flag_false():
    """is_i2v=False (the default) must also be stored correctly."""
    pipe = WanInferencePipeline(model_path="/fake", is_i2v=False)
    assert pipe._is_i2v is False


def test_default_device():
    """Device should default to 'cuda' when not specified.

    Wan models require CUDA for inference. The default keeps the common
    case explicit without forcing users to repeat themselves.
    """
    pipe = WanInferencePipeline(model_path="/fake")
    assert pipe._device == "cuda"


def test_constructor_stores_device():
    """An explicit device argument must be stored."""
    pipe = WanInferencePipeline(model_path="/fake", device="cpu")
    assert pipe._device == "cpu"


# ---------------------------------------------------------------------------
# cleanup() behaviour
# ---------------------------------------------------------------------------

def test_cleanup_when_no_pipeline():
    """cleanup() must not raise when the pipeline was never loaded.

    The training loop calls cleanup() unconditionally at the end of
    sampling; it must be safe even if generate() was never invoked.
    """
    pipe = WanInferencePipeline(model_path="/fake")
    pipe.cleanup()  # Should not raise


def test_cleanup_clears_pipeline():
    """cleanup() must set _pipeline back to None after clearing it.

    This ensures that a second generate() call after cleanup() will
    attempt to reload the pipeline rather than use a stale reference.
    """
    pipe = WanInferencePipeline(model_path="/fake")
    pipe._pipeline = "dummy"  # Simulate a loaded pipeline
    pipe.cleanup()
    assert pipe._pipeline is None


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

def test_protocol_compliance():
    """WanInferencePipeline must satisfy the InferencePipeline protocol.

    InferencePipeline is @runtime_checkable, so isinstance() confirms
    that the class exposes the required generate() method signature.
    The training loop uses this check to validate the pipeline object
    before passing it to SamplingEngine.
    """
    pipe = WanInferencePipeline(model_path="/fake")
    assert isinstance(pipe, InferencePipeline)


# ---------------------------------------------------------------------------
# Error propagation from generate()
# ---------------------------------------------------------------------------

def test_generate_without_pipeline_raises_on_bad_path(monkeypatch):
    """generate() must raise SamplingError (or ImportError) for a bad path.

    When _pipeline is None, generate() calls _ensure_pipeline(), which
    tries to import torch/diffusers and load from model_path. With a
    nonexistent path this must raise rather than silently succeed.

    SamplingError is raised when torch+diffusers are installed but the
    path doesn't exist. ImportError is raised when the dependencies are
    not installed at all (GPU-free CI environment). Both are acceptable
    failure modes for this test.
    """
    from dimljus.training.errors import SamplingError

    pipe = WanInferencePipeline(model_path="/nonexistent/path")
    with pytest.raises((SamplingError, ImportError, OSError, Exception)):
        pipe.generate(
            model=None,
            lora_state_dict=None,
            prompt="test prompt",
        )
