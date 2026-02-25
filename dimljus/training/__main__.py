"""CLI for the Dimljus training pipeline.

Commands:
    train       Run training from a config file
    plan        Print the resolved training plan without training (dry run)

Usage::

    # Full training run
    python -m dimljus.training train --config path/to/train.yaml

    # Dry run — resolve phases and print plan
    python -m dimljus.training plan --config path/to/train.yaml
"""

from __future__ import annotations

import argparse
import sys


def cmd_train(args: argparse.Namespace) -> None:
    """Run training from a config file."""
    from dimljus.config.training_loader import load_training_config
    from dimljus.training.loop import TrainingOrchestrator

    config = load_training_config(args.config)

    # Resolve model backend from config variant
    backend = _resolve_backend(config)

    # Optionally create inference pipeline for sampling
    inference_pipeline = None
    if config.sampling.enabled:
        inference_pipeline = _resolve_inference_pipeline(config)

    orchestrator = TrainingOrchestrator(
        config=config,
        model_backend=backend,
        inference_pipeline=inference_pipeline,
    )
    orchestrator.run(dry_run=args.dry_run)


def cmd_plan(args: argparse.Namespace) -> None:
    """Print the resolved training plan without training."""
    from dimljus.config.training_loader import load_training_config
    from dimljus.training.loop import TrainingOrchestrator

    config = load_training_config(args.config)

    # Plan mode uses stub backend — no GPU needed
    orchestrator = TrainingOrchestrator(
        config=config,
        model_backend=_StubBackend(),
        inference_pipeline=None,
    )
    orchestrator.run(dry_run=True)


def _resolve_backend(config: object) -> object:
    """Resolve the model backend from config.

    Tries to load the real Wan backend. Falls back to stub if the
    required packages are not installed or the variant is unknown.

    Args:
        config: DimljusTrainingConfig instance.

    Returns:
        ModelBackend implementation (WanModelBackend or _StubBackend).
    """
    model = getattr(config, "model", None)
    variant = getattr(model, "variant", None) if model else None

    if variant is None:
        print(
            "Warning: No model variant set. Using stub backend.\n"
            "Set model.variant in your config (e.g. '2.2_t2v')."
        )
        return _StubBackend()

    try:
        from dimljus.training.wan.registry import get_wan_backend
        print(f"Loading Wan model backend for variant '{variant}'...")
        return get_wan_backend(config)
    except ImportError as e:
        print(
            f"Warning: Cannot load Wan backend ({e}).\n"
            f"Install with: pip install 'dimljus[wan]'\n"
            f"Falling back to stub backend."
        )
        return _StubBackend()
    except ValueError as e:
        print(f"Warning: {e}\nFalling back to stub backend.")
        return _StubBackend()


def _resolve_inference_pipeline(config: object) -> object | None:
    """Create inference pipeline for sampling during training.

    Args:
        config: DimljusTrainingConfig instance.

    Returns:
        WanInferencePipeline or None if not available.
    """
    model = getattr(config, "model", None)
    if model is None:
        return None

    try:
        from dimljus.training.wan.inference import WanInferencePipeline

        is_i2v = getattr(model, "variant", "") == "2.2_i2v"
        dtype = getattr(
            getattr(config, "training", None),
            "mixed_precision",
            "bf16",
        )

        return WanInferencePipeline(
            model_path=model.path,
            is_i2v=is_i2v,
            dtype=dtype,
        )
    except ImportError:
        return None


class _StubBackend:
    """Minimal stub backend for dry-run and plan commands.

    Satisfies the ModelBackend protocol with no-ops so the orchestrator
    can resolve phases and print plans without a real model.
    """

    @property
    def model_id(self) -> str:
        return "stub"

    @property
    def supports_moe(self) -> bool:
        return True

    @property
    def supports_reference_image(self) -> bool:
        return False

    def load_model(self, config):
        return None

    def get_lora_target_modules(self) -> list[str]:
        return []

    def get_expert_mask(self, timesteps, boundary_ratio):
        return (None, None)

    def prepare_model_inputs(self, batch, timesteps, noisy_latents):
        return {}

    def forward(self, model, **inputs):
        return None

    def setup_gradient_checkpointing(self, model):
        pass

    def get_noise_schedule(self):
        from dimljus.training.noise import FlowMatchingSchedule
        return FlowMatchingSchedule()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m dimljus.training",
        description="Dimljus training pipeline — video LoRA training with differential MoE.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ─── train ───
    train_parser = subparsers.add_parser(
        "train",
        help="Run training from a config file.",
    )
    train_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the Dimljus training config YAML.",
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve phases and print plan without training.",
    )

    # ─── plan ───
    plan_parser = subparsers.add_parser(
        "plan",
        help="Print resolved training plan (dry run).",
    )
    plan_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the Dimljus training config YAML.",
    )

    return parser


def main() -> None:
    """Entry point for the training CLI."""
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "train": cmd_train,
        "plan": cmd_plan,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nTraining interrupted.", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
