"""Tests for dimljus.training.loop — training orchestrator."""

import numpy as np
import pytest

from dimljus.training.loop import TrainingOrchestrator
from dimljus.training.noise import FlowMatchingSchedule
from dimljus.training.phase import PhaseType


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------

class MockModelBackend:
    """Minimal model backend for orchestrator testing."""

    @property
    def model_id(self):
        return "mock"

    @property
    def supports_moe(self):
        return True

    @property
    def supports_reference_image(self):
        return False

    def load_model(self, config):
        return "mock_model"

    def get_lora_target_modules(self):
        return ["to_q", "to_k"]

    def get_expert_mask(self, timesteps, boundary_ratio):
        return (np.ones(1), np.zeros(1))

    def prepare_model_inputs(self, batch, timesteps, noisy_latents):
        return {}

    def forward(self, model, **inputs):
        return 0.1  # Mock loss

    def setup_gradient_checkpointing(self, model):
        pass

    def get_noise_schedule(self):
        return FlowMatchingSchedule(1000)


class MockSaveConfig:
    output_dir = ""
    name = "test_lora"
    save_every_n_epochs = 5
    save_last = True
    max_checkpoints = None
    format = "safetensors"

class MockLoggingConfig:
    backends = ["console"]
    log_every_n_steps = 100
    wandb_project = None
    wandb_entity = None
    wandb_run_name = None
    wandb_group = None
    wandb_tags = []
    vram_sample_every_n_steps = 50

class MockSamplingConfig:
    enabled = False
    every_n_epochs = 5
    prompts = []
    neg = ""
    seed = 42
    walk_seed = True
    sample_steps = 30
    guidance_scale = 5.0
    sample_dir = None
    skip_phases = []

class MockOptimizer:
    type = "adamw8bit"
    learning_rate = 5e-5
    weight_decay = 0.01

class MockScheduler:
    type = "cosine_with_min_lr"
    warmup_steps = 0
    min_lr_ratio = 0.01

class MockLora:
    rank = 16
    alpha = 16
    dropout = 0.0

class MockExpertOverrides:
    def __init__(self, **kwargs):
        self.enabled = kwargs.get("enabled", True)
        self.learning_rate = None
        self.dropout = None
        self.max_epochs = kwargs.get("max_epochs", 30)
        self.fork_targets = None
        self.block_targets = None
        self.resume_from = None
        self.batch_size = None
        self.gradient_accumulation_steps = None
        self.caption_dropout_rate = None
        self.weight_decay = None
        self.min_lr_ratio = None
        self.optimizer_type = None
        self.scheduler_type = None

class MockMoeConfig:
    def __init__(self, **kwargs):
        self.enabled = kwargs.get("enabled", True)
        self.fork_enabled = kwargs.get("fork_enabled", True)
        self.expert_order = ["low_noise", "high_noise"]
        self.high_noise = MockExpertOverrides(max_epochs=30)
        self.low_noise = MockExpertOverrides(max_epochs=50)
        self.boundary_ratio = None

class MockTrainingConfig:
    def __init__(self, **kwargs):
        self.unified_epochs = kwargs.get("unified_epochs", 10)
        self.batch_size = 1
        self.gradient_accumulation_steps = 1
        self.caption_dropout_rate = 0.1
        self.unified_targets = None
        self.unified_block_targets = None
        self.resume_from = None
        self.gradient_checkpointing = False
        self.timestep_sampling = "uniform"

class MockModelConfig:
    boundary_ratio = 0.875
    flow_shift = 3.0
    variant = "2.2_t2v"

class MockConfig:
    def __init__(self, tmp_path, **kwargs):
        self.training = kwargs.get("training", MockTrainingConfig())
        self.optimizer = MockOptimizer()
        self.scheduler = MockScheduler()
        self.lora = MockLora()
        self.moe = kwargs.get("moe", MockMoeConfig())
        self.model = MockModelConfig()
        save = MockSaveConfig()
        save.output_dir = str(tmp_path / "output")
        self.save = save
        self.logging = MockLoggingConfig()
        sampling = MockSamplingConfig()
        sampling.sample_dir = str(tmp_path / "samples")
        self.sampling = sampling

    def model_dump(self):
        """Mimic Pydantic model_dump() for config save testing."""
        return {"model": {"variant": "2.2_t2v"}, "test": True}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrchestratorInit:
    """Orchestrator initialization and phase resolution."""

    def test_resolves_phases(self, tmp_path):
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        assert len(orch.phases) == 3  # unified + low_noise + high_noise

    def test_unified_only(self, tmp_path):
        config = MockConfig(tmp_path, moe=MockMoeConfig(fork_enabled=False))
        orch = TrainingOrchestrator(config, MockModelBackend())
        assert len(orch.phases) == 1
        assert orch.phases[0].phase_type == PhaseType.UNIFIED

    def test_initial_step_zero(self, tmp_path):
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        assert orch.global_step == 0


class TestDryRun:
    """Dry run mode — resolve and print without training."""

    def test_dry_run_no_error(self, tmp_path, capsys):
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        orch.run(dry_run=True)
        output = capsys.readouterr().out
        assert "TRAINING PLAN" in output

    def test_dry_run_shows_phases(self, tmp_path, capsys):
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        orch.run(dry_run=True)
        output = capsys.readouterr().out
        assert "UNIFIED" in output

    def test_dry_run_no_checkpoints(self, tmp_path):
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        orch.run(dry_run=True)
        output_dir = tmp_path / "output"
        # Dry run should NOT create output directories
        assert not output_dir.exists() or not any(output_dir.iterdir())


class TestTrainingRun:
    """Full training run with mock backend."""

    def test_creates_output_dirs(self, tmp_path):
        config = MockConfig(tmp_path, moe=MockMoeConfig(fork_enabled=False))
        config.training.unified_epochs = 2
        config.save.save_every_n_epochs = 1
        orch = TrainingOrchestrator(config, MockModelBackend())
        orch.run(dataset=None)
        assert (tmp_path / "output" / "unified").is_dir()

    def test_saves_training_state(self, tmp_path):
        config = MockConfig(tmp_path, moe=MockMoeConfig(fork_enabled=False))
        config.training.unified_epochs = 2
        config.save.save_every_n_epochs = 1
        orch = TrainingOrchestrator(config, MockModelBackend())
        orch.run(dataset=None)
        state_path = tmp_path / "output" / "training_state.json"
        assert state_path.is_file()


class TestMetricsInfrastructure:
    """Verify orchestrator creates VRAMTracker, RunTimer, and related infra."""

    def test_has_vram_tracker(self, tmp_path):
        """Orchestrator creates a VRAMTracker instance."""
        from dimljus.training.vram import VRAMTracker
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        assert hasattr(orch, "_vram_tracker")
        assert isinstance(orch._vram_tracker, VRAMTracker)

    def test_has_run_timer(self, tmp_path):
        """Orchestrator creates a RunTimer instance."""
        from dimljus.training.metrics import RunTimer
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        assert hasattr(orch, "_timer")
        assert isinstance(orch._timer, RunTimer)

    def test_run_summary_called(self, tmp_path, capsys):
        """run() prints a training complete summary at the end."""
        config = MockConfig(tmp_path, moe=MockMoeConfig(fork_enabled=False))
        config.training.unified_epochs = 1
        config.save.save_every_n_epochs = 1
        orch = TrainingOrchestrator(config, MockModelBackend())
        orch.run(dataset=None)
        output = capsys.readouterr().out
        assert "TRAINING COMPLETE" in output

    def test_resolved_config_saved(self, tmp_path):
        """run() saves resolved_config.yaml to the output directory."""
        config = MockConfig(tmp_path, moe=MockMoeConfig(fork_enabled=False))
        config.training.unified_epochs = 1
        config.save.save_every_n_epochs = 1
        orch = TrainingOrchestrator(config, MockModelBackend())
        orch.run(dataset=None)
        config_path = tmp_path / "output" / "resolved_config.yaml"
        assert config_path.is_file()

    def test_vram_tracker_interval_from_config(self, tmp_path):
        """VRAMTracker uses the interval from config.logging."""
        config = MockConfig(tmp_path)
        config.logging.vram_sample_every_n_steps = 100
        orch = TrainingOrchestrator(config, MockModelBackend())
        assert orch._vram_tracker._sample_interval == 100


class TestPhaseTransitions:
    """Phase transitions and fork mechanism."""

    def test_fork_logged(self, tmp_path, capsys):
        config = MockConfig(tmp_path)
        config.training.unified_epochs = 1
        config.save.save_every_n_epochs = 1
        orch = TrainingOrchestrator(config, MockModelBackend())
        orch.run(dataset=None)
        output = capsys.readouterr().out
        # Should show fork message after unified phase
        assert "FORK" in output or "unified" in output.lower()


# ---------------------------------------------------------------------------
# Tests: WeightVerifier wiring (Plan 02-02)
# ---------------------------------------------------------------------------

class TestWeightVerifierWiring:
    """Verify WeightVerifier is created and wired into the orchestrator."""

    def test_has_weight_verifier(self, tmp_path):
        """Orchestrator creates a WeightVerifier instance."""
        from dimljus.training.verification import WeightVerifier
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        assert hasattr(orch, "_weight_verifier")
        assert isinstance(orch._weight_verifier, WeightVerifier)

    def test_has_frozen_results_dict(self, tmp_path):
        """Orchestrator initializes an empty frozen_results dict."""
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        assert hasattr(orch, "_frozen_results")
        assert isinstance(orch._frozen_results, dict)
        assert len(orch._frozen_results) == 0


class TestGetFrozenExpertName:
    """Test _get_frozen_expert_name helper method."""

    def _make_phase(self, **kwargs):
        """Create a minimal TrainingPhase for testing."""
        from dimljus.training.phase import TrainingPhase
        defaults = dict(
            phase_type=PhaseType.UNIFIED, max_epochs=10,
            learning_rate=5e-5, weight_decay=0.01,
            optimizer_type="adamw8bit", scheduler_type="cosine_with_min_lr",
            min_lr_ratio=0.01, warmup_steps=0, batch_size=1,
            gradient_accumulation_steps=1, caption_dropout_rate=0.1,
            lora_dropout=0.0, fork_targets=None, block_targets=None,
            resume_from=None, boundary_ratio=None, active_expert=None,
        )
        defaults.update(kwargs)
        return TrainingPhase(**defaults)

    def test_high_noise_phase_freezes_low_noise(self, tmp_path):
        """During high_noise training, low_noise should be frozen."""
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        phase = self._make_phase(
            phase_type=PhaseType.HIGH_NOISE,
            active_expert="high_noise",
        )
        assert orch._get_frozen_expert_name(phase) == "low_noise"

    def test_low_noise_phase_freezes_high_noise(self, tmp_path):
        """During low_noise training, high_noise should be frozen."""
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        phase = self._make_phase(
            phase_type=PhaseType.LOW_NOISE,
            active_expert="low_noise",
        )
        assert orch._get_frozen_expert_name(phase) == "high_noise"

    def test_unified_phase_no_frozen_expert(self, tmp_path):
        """Unified phase has no frozen expert."""
        config = MockConfig(tmp_path)
        orch = TrainingOrchestrator(config, MockModelBackend())
        phase = self._make_phase(phase_type=PhaseType.UNIFIED)
        assert orch._get_frozen_expert_name(phase) is None


class TestRunSummaryWithFrozenChecks:
    """Verify frozen checks are included in end-of-run summary."""

    def test_summary_includes_frozen_checks_when_populated(self, tmp_path, capsys):
        """When frozen_results has entries, they appear in the summary."""
        config = MockConfig(tmp_path, moe=MockMoeConfig(fork_enabled=False))
        config.training.unified_epochs = 1
        config.save.save_every_n_epochs = 1
        orch = TrainingOrchestrator(config, MockModelBackend())

        # Manually populate frozen results to simulate a real run
        orch._frozen_results = {"high_noise": True, "low_noise": True}
        orch.run(dataset=None)

        output = capsys.readouterr().out
        assert "TRAINING COMPLETE" in output
        # The frozen checks should appear in summary since we populated them
        assert "Frozen expert verification" in output
        assert "PASS" in output

    def test_summary_no_frozen_section_when_empty(self, tmp_path, capsys):
        """When no frozen results, the summary section is absent."""
        config = MockConfig(tmp_path, moe=MockMoeConfig(fork_enabled=False))
        config.training.unified_epochs = 1
        config.save.save_every_n_epochs = 1
        orch = TrainingOrchestrator(config, MockModelBackend())
        orch.run(dataset=None)

        output = capsys.readouterr().out
        assert "TRAINING COMPLETE" in output
        # No frozen checks section when results dict is empty
        assert "Frozen expert verification" not in output
