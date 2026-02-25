"""Multi-backend training logger with phase awareness.

Provides consistent logging across console, TensorBoard, and W&B.
Each backend is optional — missing libraries are handled gracefully.

Key features:
    - Phase-prefixed logging (unified/, high_noise/, low_noise/)
    - Training plan pretty-print before training starts
    - Phase transition events (start, end, fork)
    - Progress bars for console output
    - GPU-free (mocked backends for testing)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from dimljus.training.phase import PhaseType, TrainingPhase


class TrainingLogger:
    """Multi-backend logger for the training loop.

    Initializes requested backends and dispatches log calls to all active
    ones. Console is always available; TensorBoard and W&B are optional.

    Args:
        backends: List of backend names ('console', 'tensorboard', 'wandb').
        output_dir: Directory for TensorBoard logs.
        wandb_project: W&B project name (required if 'wandb' in backends).
        wandb_entity: W&B team/org name.
        wandb_run_name: W&B run name.
        log_every_n_steps: How often to log metrics.
    """

    def __init__(
        self,
        backends: list[str] | None = None,
        output_dir: str | Path | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        log_every_n_steps: int = 10,
    ) -> None:
        self._backends: list[str] = backends or ["console"]
        self._output_dir = Path(output_dir) if output_dir else None
        self._log_every_n_steps = log_every_n_steps
        self._step_count = 0

        # Initialize backends
        self._tb_writer: Any = None
        self._wandb_run: Any = None

        if "tensorboard" in self._backends and self._output_dir is not None:
            self._init_tensorboard()

        if "wandb" in self._backends:
            self._init_wandb(
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name,
            )

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard writer (optional import)."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self._output_dir / "tensorboard"  # type: ignore[operator]
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
        except ImportError:
            print(
                "Warning: tensorboard not installed. "
                "Install with: pip install tensorboard",
                file=sys.stderr,
            )

    def _init_wandb(
        self,
        project: str | None,
        entity: str | None,
        run_name: str | None,
    ) -> None:
        """Initialize W&B run (optional import)."""
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=project or "dimljus",
                entity=entity,
                name=run_name,
                reinit=True,
            )
        except ImportError:
            print(
                "Warning: wandb not installed. "
                "Install with: pip install wandb",
                file=sys.stderr,
            )

    def print_training_plan(self, phases: list[TrainingPhase]) -> None:
        """Pretty-print the resolved training plan before training starts.

        Shows each phase with its key parameters so the user can review
        the setup before committing GPU time.

        Args:
            phases: List of resolved TrainingPhase objects.
        """
        print("\n" + "=" * 60)
        print("  DIMLJUS TRAINING PLAN")
        print("=" * 60)

        for i, phase in enumerate(phases):
            phase_label = phase.phase_type.value.upper()
            if phase.active_expert:
                phase_label += f" (expert: {phase.active_expert})"

            print(f"\n  Phase {i + 1}: {phase_label}")
            print(f"  {'─' * 40}")
            print(f"    Epochs:        {phase.max_epochs}")
            print(f"    Learning rate:  {phase.learning_rate:.2e}")
            print(f"    Optimizer:      {phase.optimizer_type}")
            print(f"    Scheduler:      {phase.scheduler_type}")
            print(f"    Batch size:     {phase.batch_size}")
            print(f"    Grad accum:     {phase.gradient_accumulation_steps}")
            print(f"    Caption dropout: {phase.caption_dropout_rate:.1%}")
            print(f"    LoRA dropout:   {phase.lora_dropout:.1%}")

            if phase.fork_targets:
                print(f"    Fork targets:   {', '.join(phase.fork_targets)}")
            if phase.block_targets:
                print(f"    Block targets:  {phase.block_targets}")
            if phase.boundary_ratio is not None:
                print(f"    Boundary ratio: {phase.boundary_ratio}")
            if phase.resume_from:
                print(f"    Resume from:    {phase.resume_from}")

        print(f"\n  Total phases: {len(phases)}")
        total_epochs = sum(p.max_epochs for p in phases)
        print(f"  Total epochs: {total_epochs}")
        print("=" * 60 + "\n")

    def log_phase_start(self, phase: TrainingPhase, phase_index: int) -> None:
        """Log the start of a training phase.

        Args:
            phase: The phase starting.
            phase_index: 0-based index into the phase list.
        """
        label = phase.phase_type.value.upper()
        msg = f"Starting phase {phase_index + 1}: {label} ({phase.max_epochs} epochs)"

        if "console" in self._backends:
            print(f"\n{'─' * 50}")
            print(f"  {msg}")
            print(f"  LR: {phase.learning_rate:.2e} | Optimizer: {phase.optimizer_type}")
            print(f"{'─' * 50}")

    def log_phase_end(self, phase: TrainingPhase, phase_index: int) -> None:
        """Log the end of a training phase.

        Args:
            phase: The phase ending.
            phase_index: 0-based index into the phase list.
        """
        label = phase.phase_type.value.upper()
        if "console" in self._backends:
            print(f"\n  Phase {phase_index + 1} ({label}) complete.")

    def log_fork(self) -> None:
        """Log the fork event (unified → expert phases)."""
        if "console" in self._backends:
            print(f"\n{'━' * 50}")
            print("  FORK: Splitting unified LoRA into expert copies")
            print(f"{'━' * 50}")

    def log_step(
        self,
        metrics: dict[str, float],
        global_step: int,
        phase_type: PhaseType | None = None,
    ) -> None:
        """Log training metrics for one step.

        Respects log_every_n_steps — only actually logs at the configured
        interval. All backends receive the same metrics dict.

        Args:
            metrics: Flat dict of metric names → values.
            global_step: Global step counter.
            phase_type: Optional phase for prefixing.
        """
        self._step_count = global_step

        if global_step % self._log_every_n_steps != 0:
            return

        # Prefix metrics with phase name if provided
        if phase_type is not None:
            prefix = f"{phase_type.value}/"
            prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}
        else:
            prefixed = metrics

        # Console
        if "console" in self._backends:
            loss = metrics.get("loss_ema", metrics.get("loss_raw", 0.0))
            lr = metrics.get("learning_rate", 0.0)
            epoch = metrics.get("epoch", 0)
            print(
                f"  step {global_step:>6d} | "
                f"epoch {int(epoch):>3d} | "
                f"loss {loss:.4f} | "
                f"lr {lr:.2e}"
            )

        # TensorBoard
        if self._tb_writer is not None:
            for key, value in prefixed.items():
                self._tb_writer.add_scalar(key, value, global_step)

        # W&B
        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log(prefixed, step=global_step)
            except Exception:
                pass  # Don't crash training for logging failures

    def log_checkpoint_saved(self, path: Path, phase_type: PhaseType, epoch: int) -> None:
        """Log a checkpoint save event.

        Args:
            path: Where the checkpoint was saved.
            phase_type: Phase that produced the checkpoint.
            epoch: Epoch number.
        """
        if "console" in self._backends:
            print(f"  Checkpoint saved: {path.name} (epoch {epoch})")

    def log_sample_generated(self, path: Path, prompt_index: int) -> None:
        """Log a sample generation event.

        Args:
            path: Where the sample was saved.
            prompt_index: Index of the prompt that generated this sample.
        """
        if "console" in self._backends:
            print(f"  Sample generated: {path.name} (prompt {prompt_index})")

    def close(self) -> None:
        """Close all logging backends and flush pending writes."""
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass

        if self._wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
