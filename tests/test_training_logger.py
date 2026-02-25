"""Tests for dimljus.training.logger — multi-backend logging."""

import pytest

from dimljus.training.logger import TrainingLogger
from dimljus.training.phase import PhaseType, TrainingPhase


def _make_phase(**kwargs):
    """Create a minimal TrainingPhase for testing."""
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


class TestTrainingLogger:
    """Console logger basics."""

    def test_create_console_only(self):
        logger = TrainingLogger(backends=["console"])
        assert logger is not None

    def test_print_training_plan(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phases = [
            _make_phase(),
            _make_phase(phase_type=PhaseType.HIGH_NOISE, active_expert="high_noise"),
        ]
        logger.print_training_plan(phases)
        output = capsys.readouterr().out
        assert "TRAINING PLAN" in output
        assert "UNIFIED" in output
        assert "HIGH_NOISE" in output
        assert "Total phases: 2" in output

    def test_log_phase_start(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phase = _make_phase()
        logger.log_phase_start(phase, phase_index=0)
        output = capsys.readouterr().out
        assert "UNIFIED" in output
        assert "10 epochs" in output

    def test_log_phase_end(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phase = _make_phase()
        logger.log_phase_end(phase, phase_index=0)
        output = capsys.readouterr().out
        assert "complete" in output

    def test_log_fork(self, capsys):
        logger = TrainingLogger(backends=["console"])
        logger.log_fork()
        output = capsys.readouterr().out
        assert "FORK" in output

    def test_log_step_at_interval(self, capsys):
        logger = TrainingLogger(backends=["console"], log_every_n_steps=10)
        logger.log_step(
            metrics={"loss_ema": 0.05, "learning_rate": 5e-5, "epoch": 1},
            global_step=10,
        )
        output = capsys.readouterr().out
        assert "0.0500" in output

    def test_log_step_skipped(self, capsys):
        logger = TrainingLogger(backends=["console"], log_every_n_steps=10)
        logger.log_step(
            metrics={"loss_ema": 0.05},
            global_step=7,
        )
        output = capsys.readouterr().out
        assert output == ""

    def test_log_checkpoint_saved(self, capsys, tmp_path):
        logger = TrainingLogger(backends=["console"])
        path = tmp_path / "checkpoint.safetensors"
        logger.log_checkpoint_saved(path, PhaseType.UNIFIED, epoch=5)
        output = capsys.readouterr().out
        assert "checkpoint.safetensors" in output

    def test_log_sample_generated(self, capsys, tmp_path):
        logger = TrainingLogger(backends=["console"])
        path = tmp_path / "sample.mp4"
        logger.log_sample_generated(path, prompt_index=0)
        output = capsys.readouterr().out
        assert "sample.mp4" in output

    def test_close_is_safe(self):
        logger = TrainingLogger(backends=["console"])
        logger.close()  # Should not raise

    def test_plan_shows_fork_targets(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phases = [_make_phase(fork_targets=["ffn", "self_attn"])]
        logger.print_training_plan(phases)
        output = capsys.readouterr().out
        assert "ffn" in output

    def test_plan_shows_block_targets(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phases = [_make_phase(block_targets="0-11")]
        logger.print_training_plan(phases)
        output = capsys.readouterr().out
        assert "0-11" in output

    def test_plan_shows_boundary_ratio(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phases = [_make_phase(boundary_ratio=0.875)]
        logger.print_training_plan(phases)
        output = capsys.readouterr().out
        assert "0.875" in output
