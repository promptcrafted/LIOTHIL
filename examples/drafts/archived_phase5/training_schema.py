"""Pydantic v2 models for the Dimljus training config.

This is the schema that describes a training run to Dimljus. It answers:
"What model am I training, how should the LoRA be configured, and what
hyperparameters should I use?"

The training config points at a data config (dimljus_data.yaml) for the
dataset definition. It does NOT load or validate the data config — that
happens at training time when both configs are needed together.

Three tiers of complexity:
  - New user: template + model path + data_config (everything else defaulted)
  - Standard: rank, learning rate, optimizer, epochs, save settings
  - Internal: every field, MoE overrides, fork-and-specialize, sampling

Model templates (wan21_t2v, wan22_t2v, wan22_i2v) provide sensible defaults
for each Wan variant. User config is deep-merged on top — user always wins.
"""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from dimljus.config.training_defaults import (
    DEFAULT_BASE_PRECISION,
    DEFAULT_BETAS,
    DEFAULT_CAPTION_DROPOUT_RATE,
    DEFAULT_CHECKPOINT_FORMAT,
    DEFAULT_EPS,
    DEFAULT_GRADIENT_ACCUMULATION,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_LORAPLUS_LR_RATIO,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MIN_LR_RATIO,
    DEFAULT_MIXED_PRECISION,
    DEFAULT_OPTIMIZER,
    DEFAULT_SAMPLING_GUIDANCE,
    DEFAULT_SAMPLING_SEED,
    DEFAULT_SAMPLING_STEPS,
    DEFAULT_SAVE_EVERY_N_EPOCHS,
    DEFAULT_SCHEDULER,
    DEFAULT_TIMESTEP_SAMPLING,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_BATCH_SIZE,
    VALID_BASE_PRECISION,
    VALID_CHECKPOINT_FORMATS,
    VALID_LOG_BACKENDS,
    VALID_MIXED_PRECISION,
    VALID_MODEL_TEMPLATES,
    VALID_OPTIMIZERS,
    VALID_SCHEDULERS,
    VALID_TIMESTEP_SAMPLING,
)


# ─── Model Config ───


class ModelConfig(BaseModel):
    """What model to train and where to find it.

    The template field selects a set of model-specific defaults (architecture
    info, channel counts, MoE settings). User fields override template values.

    The path can be a local directory or a HuggingFace model ID (org/model).
    Local paths are resolved relative to the config file's location.
    """

    template: str | None = Field(
        default=None,
        description=(
            "Model template name. Provides sensible defaults for the model variant. "
            "Valid: wan21_t2v, wan22_t2v, wan22_i2v. "
            "Set to null to configure everything manually."
        ),
    )
    path: str = Field(
        description=(
            "Path to model weights. Can be a local directory or a HuggingFace ID "
            "(e.g. 'Wan-AI/Wan2.2-T2V-14B-Diffusers'). Local paths are resolved "
            "relative to the config file's location."
        ),
    )
    family: str | None = Field(
        default=None,
        description=(
            "Model family name (e.g. 'wan'). Auto-set by template. "
            "Override only if using a custom model not covered by templates."
        ),
    )
    variant: str | None = Field(
        default=None,
        description=(
            "Model variant (e.g. '2.2_t2v'). Auto-set by template. "
            "Override only if using a custom model."
        ),
    )
    is_moe: bool | None = Field(
        default=None,
        description=(
            "Whether this model uses Mixture of Experts architecture. "
            "Auto-set by template. Wan 2.2 = True, Wan 2.1 = False."
        ),
    )
    in_channels: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Input channel count for the transformer. Auto-set by template. "
            "T2V = 16 (latent only), I2V = 36 (latent + VAE-encoded reference)."
        ),
    )
    num_layers: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Number of transformer blocks. Auto-set by template. "
            "Wan models use 40 layers."
        ),
    )
    boundary_ratio: float | None = Field(
        default=None,
        description=(
            "MoE expert routing boundary (SNR threshold). "
            "T2V: 0.875, I2V: 0.900. Only relevant for MoE models."
        ),
    )
    flow_shift: float | None = Field(
        default=None,
        description=(
            "Flow matching shift parameter. Controls the noise schedule curve. "
            "480p: 3.0, 720p: 5.0. Auto-set by template."
        ),
    )
    num_train_timesteps: int = Field(
        default=1000,
        gt=0,
        description="Total noise schedule steps. 1000 for all Wan models.",
    )

    @field_validator("template", mode="before")
    @classmethod
    def validate_template(cls, v: str | None) -> str | None:
        """Validate template against known model templates."""
        if v is not None and v not in VALID_MODEL_TEMPLATES:
            valid_list = ", ".join(sorted(VALID_MODEL_TEMPLATES))
            raise ValueError(
                f"Unknown model template '{v}'. "
                f"Valid templates: {valid_list}. "
                f"Set to null to configure the model manually."
            )
        return v


# ─── LoRA Config ───


class LoraConfig(BaseModel):
    """LoRA adapter settings (base settings, before MoE expert overrides).

    These are the defaults that apply to all LoRA layers. MoE expert
    overrides (in MoeConfig) can override rank, alpha, and target modules
    per expert — anything not overridden inherits from here.
    """

    rank: int = Field(
        default=DEFAULT_LORA_RANK,
        gt=0,
        description=(
            "LoRA rank (dimension of low-rank decomposition). Higher = more "
            "capacity but more VRAM and risk of overfitting. "
            "32 is the community standard for Wan 14B."
        ),
    )
    alpha: int = Field(
        default=DEFAULT_LORA_ALPHA,
        gt=0,
        description=(
            "LoRA alpha (scaling factor). Effective scaling = alpha/rank. "
            "alpha = rank/2 is conservative (0.5x scaling). "
            "alpha = rank is neutral (1.0x scaling)."
        ),
    )
    dropout: float = Field(
        default=DEFAULT_LORA_DROPOUT,
        ge=0.0,
        le=1.0,
        description=(
            "LoRA dropout rate (0.0-1.0). Randomly drops LoRA outputs during "
            "training to prevent overfitting. 0.0 = no dropout. "
            "Consider 0.05 for datasets with >20 clips."
        ),
    )
    loraplus_lr_ratio: float = Field(
        default=DEFAULT_LORAPLUS_LR_RATIO,
        ge=1.0,
        description=(
            "LoRA+ B-matrix learning rate multiplier (>= 1.0). The B matrix "
            "(output projection) starts from zero and needs to learn faster. "
            "4.0 is validated on Wan models."
        ),
    )
    target_modules: list[str] | None = Field(
        default=None,
        description=(
            "Which model modules to apply LoRA to. null = use model defaults "
            "(typically all attention projections + FFN layers). "
            "Example: ['to_q', 'to_k', 'to_v', 'to_out.0']"
        ),
    )
    exclude_modules: list[str] | None = Field(
        default=None,
        description=(
            "Modules to exclude from LoRA. Applied after target_modules. "
            "Useful for freezing specific layers (e.g. exclude early blocks "
            "that handle global composition and are less subject-specific)."
        ),
    )
    block_rank_overrides: dict[str, int] | None = Field(
        default=None,
        description=(
            "Per-block rank overrides. Keys are block ranges (e.g. '0-9', '10-29'). "
            "Values are rank for those blocks. Unspecified blocks use the base rank. "
            "Example: {'0-9': 48, '30-39': 16} — more capacity early, less late."
        ),
    )
    use_mua_init: bool = Field(
        default=False,
        description=(
            "Enable muA (mu-parameterization aligned) LoRA initialization. "
            "Uses special init for the B matrix and automatically sets alpha = rank. "
            "Research-grade feature — improves scaling behavior at high ranks."
        ),
    )


# ─── Optimizer Config ───


class OptimizerConfig(BaseModel):
    """Optimizer settings for training.

    These are the base optimizer settings. MoE expert overrides can
    specify a different optimizer type or learning rate per expert.
    """

    type: str = Field(
        default=DEFAULT_OPTIMIZER,
        description=(
            "Optimizer type. adamw8bit is the community standard — cuts optimizer "
            "VRAM in half with negligible quality loss."
        ),
    )
    learning_rate: float = Field(
        default=DEFAULT_LEARNING_RATE,
        gt=0.0,
        description=(
            "Peak learning rate. 2e-4 is the community sweet spot for Wan 14B LoRA. "
            "Prodigy optimizer requires lr=1.0 (it sets the real LR automatically)."
        ),
    )
    weight_decay: float = Field(
        default=DEFAULT_WEIGHT_DECAY,
        ge=0.0,
        description="Weight decay for regularization. 0.01 is the standard default.",
    )
    betas: list[float] = Field(
        default=DEFAULT_BETAS.copy(),
        description=(
            "Momentum parameters. [beta1, beta2] for Adam variants (length 2). "
            "[beta1, beta2, beta3] for CAME (length 3). "
            "Default [0.9, 0.999] is the PyTorch standard."
        ),
    )
    eps: float | list[float] = Field(
        default=DEFAULT_EPS,
        description=(
            "Epsilon for numerical stability. Scalar for Adam variants. "
            "CAME can use a list of two epsilons [eps1, eps2]. "
            "Default 1e-8 is the PyTorch standard."
        ),
    )
    max_grad_norm: float | None = Field(
        default=DEFAULT_MAX_GRAD_NORM,
        description=(
            "Maximum gradient norm for clipping. Prevents training instability "
            "from gradient spikes. null = no clipping. 1.0 is safe default."
        ),
    )
    optimizer_args: dict = Field(
        default_factory=dict,
        description=(
            "Extra keyword arguments passed directly to the optimizer constructor. "
            "Use for optimizer-specific settings not covered by the fields above. "
            "Example: {'amsgrad': true} for AMSGrad variant of Adam."
        ),
    )

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate optimizer type against supported options."""
        if v not in VALID_OPTIMIZERS:
            valid_list = ", ".join(sorted(VALID_OPTIMIZERS))
            raise ValueError(
                f"Unknown optimizer '{v}'. "
                f"Valid optimizers: {valid_list}."
            )
        return v

    @field_validator("betas", mode="before")
    @classmethod
    def validate_betas(cls, v: list[float]) -> list[float]:
        """Betas must be a list of 2-3 floats in [0, 1)."""
        if not isinstance(v, list):
            raise ValueError(
                f"betas must be a list, got {type(v).__name__}."
            )
        if len(v) < 2 or len(v) > 3:
            raise ValueError(
                f"betas must have 2 elements (Adam) or 3 elements (CAME), "
                f"got {len(v)}."
            )
        for i, beta in enumerate(v):
            if not (0.0 <= beta < 1.0):
                raise ValueError(
                    f"betas[{i}] = {beta} is out of range. "
                    f"Each beta must be >= 0.0 and < 1.0."
                )
        return v


# ─── Scheduler Config ───


class SchedulerConfig(BaseModel):
    """Learning rate scheduler settings.

    The scheduler controls how the learning rate changes over the course
    of training. cosine_with_min_lr is the default — it decays smoothly
    but keeps a floor so late-training detail refinement still gets updates.
    """

    type: str = Field(
        default=DEFAULT_SCHEDULER,
        description=(
            "Scheduler type. cosine_with_min_lr is recommended — decays smoothly "
            "with a floor to prevent zero-LR stagnation in late training."
        ),
    )
    warmup_steps: int = Field(
        default=DEFAULT_WARMUP_STEPS,
        ge=0,
        description=(
            "Steps of linear warmup from 0 to peak LR. 0 = no warmup. "
            "Wan LoRA typically needs no warmup."
        ),
    )
    min_lr: float | None = Field(
        default=None,
        description=(
            "Absolute minimum learning rate. Takes precedence over min_lr_ratio "
            "when set. null = use min_lr_ratio instead."
        ),
    )
    min_lr_ratio: float = Field(
        default=DEFAULT_MIN_LR_RATIO,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum LR as fraction of peak LR (0.0-1.0). Used when min_lr is null. "
            "0.01 = LR floor at 1% of peak. Only applies to schedulers that decay."
        ),
    )
    rex_alpha: float = Field(
        default=0.1,
        description="Rex scheduler alpha parameter. Controls early-phase aggressiveness.",
    )
    rex_beta: float = Field(
        default=0.9,
        description="Rex scheduler beta parameter. Controls late-phase stability.",
    )

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate scheduler type against supported options."""
        if v not in VALID_SCHEDULERS:
            valid_list = ", ".join(sorted(VALID_SCHEDULERS))
            raise ValueError(
                f"Unknown scheduler '{v}'. "
                f"Valid schedulers: {valid_list}."
            )
        return v


# ─── MoE Expert Overrides ───


class MoeExpertOverrides(BaseModel):
    """Per-expert hyperparameter overrides for MoE differential training.

    All fields are optional. null means "inherit from the base config."
    This is the core Dimljus innovation: high-noise and low-noise experts
    need fundamentally different training regimes.

    High-noise expert: converges faster on coarse compositional features.
    Lower rank, higher LR, fewer epochs.

    Low-noise expert: needs longer, gentler training for fine detail.
    Higher rank, lower LR, more epochs. Overfits rapidly when trained
    aggressively (washed-out artifacts, failed dynamic expressions).
    """

    rank: int | None = Field(
        default=None,
        gt=0,
        description="LoRA rank for this expert. null = inherit from lora.rank.",
    )
    alpha: int | None = Field(
        default=None,
        gt=0,
        description="LoRA alpha for this expert. null = inherit from lora.alpha.",
    )
    learning_rate: float | None = Field(
        default=None,
        gt=0.0,
        description="Learning rate for this expert. null = inherit from optimizer.learning_rate.",
    )
    max_epochs: int | None = Field(
        default=None,
        gt=0,
        description="Max training epochs for this expert. null = inherit from training.max_epochs.",
    )
    target_modules: list[str] | None = Field(
        default=None,
        description="Target modules for this expert. null = inherit from lora.target_modules.",
    )
    exclude_modules: list[str] | None = Field(
        default=None,
        description="Excluded modules for this expert. null = inherit from lora.exclude_modules.",
    )
    block_rank_overrides: dict[str, int] | None = Field(
        default=None,
        description="Per-block rank overrides for this expert. null = inherit from lora.block_rank_overrides.",
    )
    optimizer_type: str | None = Field(
        default=None,
        description="Optimizer type for this expert. null = inherit from optimizer.type.",
    )
    scheduler_type: str | None = Field(
        default=None,
        description="Scheduler type for this expert. null = inherit from scheduler.type.",
    )

    @field_validator("optimizer_type", mode="before")
    @classmethod
    def validate_optimizer_type(cls, v: str | None) -> str | None:
        """Validate per-expert optimizer if specified."""
        if v is not None and v not in VALID_OPTIMIZERS:
            valid_list = ", ".join(sorted(VALID_OPTIMIZERS))
            raise ValueError(
                f"Unknown optimizer '{v}'. Valid optimizers: {valid_list}."
            )
        return v

    @field_validator("scheduler_type", mode="before")
    @classmethod
    def validate_scheduler_type(cls, v: str | None) -> str | None:
        """Validate per-expert scheduler if specified."""
        if v is not None and v not in VALID_SCHEDULERS:
            valid_list = ", ".join(sorted(VALID_SCHEDULERS))
            raise ValueError(
                f"Unknown scheduler '{v}'. Valid schedulers: {valid_list}."
            )
        return v


# ─── Fork and Specialize Config ───


class ForkAndSpecializeConfig(BaseModel):
    """Experimental: unified warmup → fork → per-expert specialization.

    This strategy trains a single unified LoRA first (treating both experts
    the same), then forks into separate per-expert LoRAs at a specified point.
    Based on task vector analysis showing low-noise expert IS essentially
    Wan 2.1 — specialization was one-sided during pretraining.

    The theory: unified warmup captures shared features, then forking
    lets each expert specialize without duplicating early learning.
    """

    unified_epochs: int = Field(
        default=10,
        gt=0,
        description=(
            "Number of epochs for the unified (shared) training phase. "
            "Both experts train with the same LoRA during this phase."
        ),
    )
    unified_rank: int | None = Field(
        default=None,
        gt=0,
        description=(
            "LoRA rank during the unified phase. null = use base lora.rank. "
            "May want a different rank than the per-expert phase."
        ),
    )
    freeze_shared_after_fork: bool = Field(
        default=False,
        description=(
            "Freeze shared parameters (norms, modulation) after forking. "
            "Preserves features learned during unified phase."
        ),
    )
    fork_criterion: Literal["epochs", "loss_plateau"] = Field(
        default="epochs",
        description=(
            "When to trigger the fork. 'epochs' = after unified_epochs. "
            "'loss_plateau' = when loss stops decreasing (experimental)."
        ),
    )


# ─── MoE Config ───


class MoeConfig(BaseModel):
    """Container for all Mixture of Experts settings.

    MoE support is the core differentiator of Dimljus. Wan 2.2 uses a
    dual-expert design where different noise levels are handled by
    different experts. This config controls per-expert hyperparameter
    overrides and the experimental fork-and-specialize strategy.

    When enabled=True, the trainer routes training samples to the
    appropriate expert based on the noise level and boundary_ratio.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Enable MoE differential training. Auto-set True for MoE templates "
            "(wan22_t2v, wan22_i2v). Must be False for non-MoE models (wan21_t2v)."
        ),
    )
    high_noise: MoeExpertOverrides = Field(
        default_factory=MoeExpertOverrides,
        description=(
            "Hyperparameter overrides for the high-noise expert. "
            "Handles early denoising — global composition and motion."
        ),
    )
    low_noise: MoeExpertOverrides = Field(
        default_factory=MoeExpertOverrides,
        description=(
            "Hyperparameter overrides for the low-noise expert. "
            "Handles late denoising — fine detail and texture."
        ),
    )
    boundary_ratio: float | None = Field(
        default=None,
        description=(
            "Override the model's default expert routing boundary. "
            "null = use model.boundary_ratio (from template or user config). "
            "Only set this if you want to experiment with different boundaries."
        ),
    )
    fork_and_specialize: ForkAndSpecializeConfig | None = Field(
        default=None,
        description=(
            "Experimental: unified warmup → fork → per-expert specialization. "
            "null = disabled (standard per-expert training from the start)."
        ),
    )


# ─── Training Config ───


class TrainingLoopConfig(BaseModel):
    """Training loop parameters.

    These control the mechanics of the training loop itself — epochs,
    batch size, precision, noise sampling. Model-specific settings
    (like MoE routing) are in their own config sections.
    """

    max_epochs: int = Field(
        default=DEFAULT_MAX_EPOCHS,
        gt=0,
        description="Maximum training epochs. 50 is enough for most Wan LoRA datasets.",
    )
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        gt=0,
        description=(
            "Training batch size. Video LoRA is VRAM-constrained — batch size 1 "
            "is the practical default. Use gradient_accumulation_steps for "
            "effective larger batches."
        ),
    )
    gradient_accumulation_steps: int = Field(
        default=DEFAULT_GRADIENT_ACCUMULATION,
        gt=0,
        description=(
            "Steps between optimizer updates. Simulates larger batch sizes "
            "without additional VRAM. Effective batch = batch_size * this value."
        ),
    )
    gradient_checkpointing: bool = Field(
        default=True,
        description=(
            "Recompute activations during backward pass to save VRAM. "
            "Always on for video training — the memory savings are essential."
        ),
    )
    mixed_precision: str = Field(
        default=DEFAULT_MIXED_PRECISION,
        description=(
            "Mixed precision mode for training computation. "
            "bf16 is recommended for modern GPUs (Ampere+)."
        ),
    )
    base_model_precision: str = Field(
        default=DEFAULT_BASE_PRECISION,
        description=(
            "Precision for the frozen base model weights. fp8 cuts VRAM roughly "
            "in half, making 14B training possible on 24 GB GPUs."
        ),
    )
    caption_dropout_rate: float = Field(
        default=DEFAULT_CAPTION_DROPOUT_RATE,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of dropping the entire caption for a sample (0.0-1.0). "
            "Forces the model to rely on visual control signals. "
            "Higher for I2V (reference image is strong conditioning)."
        ),
    )
    timestep_sampling: str = Field(
        default=DEFAULT_TIMESTEP_SAMPLING,
        description=(
            "How to sample noise timesteps during training. 'shift' matches "
            "Wan's pretraining distribution."
        ),
    )
    discrete_flow_shift: float | None = Field(
        default=None,
        description=(
            "Override the flow shift parameter. null = use model default "
            "(from template or model.flow_shift). "
            "Only set this to experiment with different noise curves."
        ),
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility. null = random seed each run.",
    )
    resume_from: str | None = Field(
        default=None,
        description=(
            "Path to a checkpoint to resume training from. "
            "null = start fresh. Relative to config file location."
        ),
    )

    @field_validator("mixed_precision", mode="before")
    @classmethod
    def validate_mixed_precision(cls, v: str) -> str:
        """Validate mixed precision mode."""
        if v not in VALID_MIXED_PRECISION:
            valid_list = ", ".join(sorted(VALID_MIXED_PRECISION))
            raise ValueError(
                f"Invalid mixed_precision '{v}'. "
                f"Valid options: {valid_list}."
            )
        return v

    @field_validator("base_model_precision", mode="before")
    @classmethod
    def validate_base_model_precision(cls, v: str) -> str:
        """Validate base model precision."""
        if v not in VALID_BASE_PRECISION:
            valid_list = ", ".join(sorted(VALID_BASE_PRECISION))
            raise ValueError(
                f"Invalid base_model_precision '{v}'. "
                f"Valid options: {valid_list}."
            )
        return v

    @field_validator("timestep_sampling", mode="before")
    @classmethod
    def validate_timestep_sampling(cls, v: str) -> str:
        """Validate timestep sampling strategy."""
        if v not in VALID_TIMESTEP_SAMPLING:
            valid_list = ", ".join(sorted(VALID_TIMESTEP_SAMPLING))
            raise ValueError(
                f"Invalid timestep_sampling '{v}'. "
                f"Valid options: {valid_list}."
            )
        return v


# ─── Save Config ───


class SaveConfig(BaseModel):
    """Checkpoint saving settings.

    Controls where and how often checkpoints are saved. The output_dir
    is resolved relative to the config file's location.
    """

    output_dir: str = Field(
        default="./output",
        description=(
            "Directory for checkpoint output. Relative to config file location. "
            "Created automatically if it doesn't exist."
        ),
    )
    name: str = Field(
        default="dimljus_lora",
        description=(
            "Base name for checkpoint files. Final filename includes epoch number: "
            "{name}_epoch{n}.safetensors"
        ),
    )
    save_every_n_epochs: int = Field(
        default=DEFAULT_SAVE_EVERY_N_EPOCHS,
        gt=0,
        description="Save a checkpoint every N epochs. 5 is a good balance.",
    )
    save_last: bool = Field(
        default=True,
        description="Always save a checkpoint at the end of training.",
    )
    max_checkpoints: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum number of checkpoints to keep. Oldest are deleted first. "
            "null = keep all checkpoints."
        ),
    )
    format: str = Field(
        default=DEFAULT_CHECKPOINT_FORMAT,
        description=(
            "Checkpoint format. safetensors is the most widely supported "
            "(ComfyUI, A1111, diffusers). diffusers = HuggingFace directory format."
        ),
    )

    @field_validator("format", mode="before")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate checkpoint format."""
        if v not in VALID_CHECKPOINT_FORMATS:
            valid_list = ", ".join(sorted(VALID_CHECKPOINT_FORMATS))
            raise ValueError(
                f"Invalid checkpoint format '{v}'. "
                f"Valid formats: {valid_list}."
            )
        return v


# ─── Logging Config ───


class LoggingConfig(BaseModel):
    """Logging and experiment tracking settings.

    Multiple backends can be active simultaneously (e.g. console + wandb).
    Console logging is always recommended — it's the only backend that
    shows progress in your terminal.
    """

    backends: list[str] = Field(
        default=["console"],
        description=(
            "Active logging backends. console = terminal output. "
            "tensorboard = local log files. wandb = Weights & Biases cloud."
        ),
    )
    log_every_n_steps: int = Field(
        default=10,
        gt=0,
        description="Log metrics every N training steps.",
    )
    wandb_project: str | None = Field(
        default=None,
        description=(
            "Weights & Biases project name. Required if 'wandb' is in backends. "
            "Example: 'dimljus-training'"
        ),
    )
    wandb_run_name: str | None = Field(
        default=None,
        description=(
            "Weights & Biases run name. null = auto-generated. "
            "Example: 'annika_r32_e50'"
        ),
    )

    @field_validator("backends", mode="before")
    @classmethod
    def validate_backends(cls, v: list[str]) -> list[str]:
        """Validate all logging backends."""
        if not isinstance(v, list):
            raise ValueError(
                f"backends must be a list, got {type(v).__name__}."
            )
        invalid = set(v) - VALID_LOG_BACKENDS
        if invalid:
            valid_list = ", ".join(sorted(VALID_LOG_BACKENDS))
            invalid_list = ", ".join(sorted(invalid))
            raise ValueError(
                f"Invalid logging backend(s): {invalid_list}. "
                f"Valid backends: {valid_list}."
            )
        return v


# ─── Sampling Config ───


class SamplingConfig(BaseModel):
    """Sample generation settings for monitoring training progress.

    When enabled, generates sample videos at regular intervals using
    the specified prompts. The same seed is used each time so you can
    visually compare quality across epochs.
    """

    enabled: bool = Field(
        default=False,
        description="Generate sample videos during training.",
    )
    every_n_epochs: int = Field(
        default=5,
        gt=0,
        description="Generate samples every N epochs.",
    )
    prompts: list[str] = Field(
        default_factory=list,
        description=(
            "Prompts to use for sample generation. One video per prompt. "
            "Empty list = no samples even if enabled=True."
        ),
    )
    seed: int = Field(
        default=DEFAULT_SAMPLING_SEED,
        description="Fixed seed for consistent comparison across epochs.",
    )
    num_inference_steps: int = Field(
        default=DEFAULT_SAMPLING_STEPS,
        gt=0,
        description="Denoising steps for sample generation.",
    )
    guidance_scale: float = Field(
        default=DEFAULT_SAMPLING_GUIDANCE,
        gt=0.0,
        description="Classifier-free guidance scale for sample generation.",
    )


# ─── Root Config ───


class DimljusTrainingConfig(BaseModel):
    """Root config model for a Dimljus training run.

    This is the complete schema for dimljus_train.yaml. Minimum viable
    config is a model template + model path + data_config path — everything
    else has sensible defaults tuned for Wan LoRA training.

    Model templates (wan21_t2v, wan22_t2v, wan22_i2v) provide defaults
    for the model variant. User config is deep-merged on top — user always wins.

    Example minimal config::

        model:
          template: wan22_t2v
          path: C:/path/to/Wan2.2-T2V-14B-Diffusers
        data_config: ./my_dataset/dimljus_data.yaml

    Example standard config::

        model:
          template: wan22_t2v
          path: C:/path/to/Wan2.2-T2V-14B-Diffusers
        data_config: ./annika/dimljus_data.yaml
        lora:
          rank: 32
          alpha: 16
        optimizer:
          learning_rate: 2e-4
        training:
          max_epochs: 50
          seed: 42
        save:
          output_dir: ./output/annika_v1
          name: annika_lora
    """

    data_config: str = Field(
        description=(
            "Path to the Dimljus data config (dimljus_data.yaml). "
            "Relative to this config file's location."
        ),
    )
    model: ModelConfig = Field(
        description="Model selection and architecture settings.",
    )
    lora: LoraConfig = Field(
        default_factory=LoraConfig,
        description="Base LoRA adapter settings.",
    )
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer settings.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning rate scheduler settings.",
    )
    moe: MoeConfig = Field(
        default_factory=MoeConfig,
        description="Mixture of Experts settings (Wan 2.2 only).",
    )
    training: TrainingLoopConfig = Field(
        default_factory=TrainingLoopConfig,
        description="Training loop parameters.",
    )
    save: SaveConfig = Field(
        default_factory=SaveConfig,
        description="Checkpoint saving settings.",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging and experiment tracking.",
    )
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="Sample video generation settings.",
    )

    @model_validator(mode="after")
    def check_moe_consistency(self) -> DimljusTrainingConfig:
        """Error if MoE is enabled on a non-MoE model.

        Wan 2.1 is a single transformer — it doesn't have experts to route
        between. Enabling MoE on it would be a config mistake.
        """
        template = self.model.template
        is_moe = self.model.is_moe

        # Check template-based inconsistency
        if template == "wan21_t2v" and self.moe.enabled:
            raise ValueError(
                "MoE is enabled but template 'wan21_t2v' is not an MoE model. "
                "Wan 2.1 uses a single transformer — there are no experts to route between. "
                "Set moe.enabled to false, or use a Wan 2.2 template (wan22_t2v, wan22_i2v)."
            )

        # Check explicit is_moe=False inconsistency
        if is_moe is False and self.moe.enabled:
            raise ValueError(
                "MoE is enabled but model.is_moe is false. "
                "Set moe.enabled to false, or set model.is_moe to true."
            )

        return self

    @model_validator(mode="after")
    def check_prodigy_lr(self) -> DimljusTrainingConfig:
        """Prodigy optimizer requires lr=1.0 — it sets the real LR automatically.

        This is a hard requirement from the Prodigy algorithm. Using any other
        LR with Prodigy produces silently wrong results.
        """
        if self.optimizer.type == "prodigy" and self.optimizer.learning_rate != 1.0:
            raise ValueError(
                f"Prodigy optimizer requires learning_rate=1.0, "
                f"got {self.optimizer.learning_rate}. "
                f"Prodigy determines the optimal learning rate automatically — "
                f"setting it to anything other than 1.0 breaks this mechanism."
            )
        return self

    @model_validator(mode="after")
    def check_wandb_project(self) -> DimljusTrainingConfig:
        """W&B backend requires a project name to know where to log."""
        if "wandb" in self.logging.backends and not self.logging.wandb_project:
            raise ValueError(
                "wandb is in logging.backends but logging.wandb_project is not set. "
                "Add a project name, e.g.:\n"
                "  logging:\n"
                "    backends: [console, wandb]\n"
                "    wandb_project: dimljus-training"
            )
        return self

    @model_validator(mode="after")
    def check_mua_alpha(self) -> DimljusTrainingConfig:
        """When muA init is enabled, alpha must equal rank.

        muA-parameterization requires alpha=rank for correct scaling behavior.
        We auto-set this rather than erroring, since it's a deterministic fix.
        """
        if self.lora.use_mua_init and self.lora.alpha != self.lora.rank:
            self.lora.alpha = self.lora.rank
        return self

    @model_validator(mode="after")
    def warn_aggressive_low_noise(self) -> DimljusTrainingConfig:
        """Warn if low-noise expert LR is aggressively high.

        Low-noise experts overfit rapidly when trained too aggressively,
        producing washed-out artifacts and failing on dynamic expressions.
        This is a soft warning, not an error — the user might know what
        they're doing.
        """
        if self.moe.enabled and self.moe.low_noise.learning_rate is not None:
            if self.moe.low_noise.learning_rate > 2e-4:
                warnings.warn(
                    f"Low-noise expert learning_rate={self.moe.low_noise.learning_rate} "
                    f"is above 2e-4. Low-noise experts overfit rapidly with high "
                    f"learning rates — this can produce washed-out artifacts and "
                    f"failed dynamic expressions. Consider using 8e-5 to 1e-4.",
                    UserWarning,
                    stacklevel=2,
                )
        return self
