"""Default values and constants for Dimljus training config.

This module defines the "ground truth" defaults that the training schema uses.
Every constant lives here so there's exactly one place to update them.

Model templates are plain dicts (not Pydantic). The loader deep-merges a
template as the base layer before applying user config on top.
"""

# ─── Model Templates ───
# Valid template names. Each maps to a dict of default settings.

VALID_MODEL_TEMPLATES: set[str] = {"wan21_t2v", "wan22_t2v", "wan22_i2v"}
"""Known model templates. The loader uses these to apply sensible defaults
for each model variant, so users only need to specify what they want to change."""


# ─── Optimizers ───

VALID_OPTIMIZERS: set[str] = {"adamw", "adamw8bit", "adafactor", "came", "prodigy"}
"""Supported optimizer types.

- adamw: standard AdamW, full precision. Best training quality, highest VRAM.
- adamw8bit: 8-bit AdamW (bitsandbytes). ~60% community usage, best validated.
- adafactor: memory-efficient, no momentum. Good for very large models.
- came: confidence-aware memory-efficient optimizer. Like Adafactor but with momentum.
- prodigy: learning-rate-free optimizer. Sets LR automatically (must use lr=1.0).
"""

DEFAULT_OPTIMIZER: str = "adamw8bit"
"""AdamW 8-bit is the community standard for Wan LoRA training.
Cuts optimizer VRAM in half with negligible quality loss."""

DEFAULT_LEARNING_RATE: float = 2e-4
"""Community sweet spot for Wan 14B LoRA training. High enough to converge
in reasonable epochs, low enough to avoid instability."""

DEFAULT_WEIGHT_DECAY: float = 0.01
"""Standard weight decay. Helps prevent overfitting without impacting convergence."""

DEFAULT_BETAS: list[float] = [0.9, 0.999]
"""Adam momentum terms. [0.9, 0.999] is the PyTorch default and works well for LoRA."""

DEFAULT_EPS: float = 1e-8
"""Adam epsilon for numerical stability. PyTorch default."""

DEFAULT_MAX_GRAD_NORM: float = 1.0
"""Gradient clipping threshold. 1.0 prevents training instability from
gradient spikes without throttling normal updates."""


# ─── Schedulers ───

VALID_SCHEDULERS: set[str] = {
    "constant",
    "cosine",
    "cosine_with_min_lr",
    "linear",
    "rex",
    "polynomial",
}
"""Supported learning rate schedulers.

- constant: fixed LR throughout. Simple, predictable.
- cosine: classic cosine decay to zero. Smooth convergence.
- cosine_with_min_lr: cosine decay with a floor. Prevents LR from going to zero.
- linear: linear decay to zero. Predictable and easy to reason about.
- rex: reciprocal-exponential schedule. Aggressive early, stable late.
- polynomial: polynomial decay. Configurable via power parameter.
"""

DEFAULT_SCHEDULER: str = "cosine_with_min_lr"
"""Cosine with a minimum LR floor prevents the learning rate from going
to zero in late epochs, which matters for video LoRA where late detail
refinement is critical."""

DEFAULT_WARMUP_STEPS: int = 0
"""Wan LoRA training typically needs no warmup. The model is already
well-conditioned from pretraining, and LoRA adapters start near zero."""

DEFAULT_MIN_LR_RATIO: float = 0.01
"""Minimum LR as a fraction of the peak LR. 1% of peak keeps the
optimizer active in late training without being so high it destabilizes."""


# ─── LoRA ───

DEFAULT_LORA_RANK: int = 32
"""Community standard rank for Wan LoRA. Enough capacity for character
identity without overfitting on typical dataset sizes (10-50 clips)."""

DEFAULT_LORA_ALPHA: int = 16
"""Alpha = rank/2 is the conservative default. Effective LR scaling
factor is alpha/rank = 0.5, preventing the adapter from dominating
the base model's behavior early in training."""

DEFAULT_LORA_DROPOUT: float = 0.0
"""No LoRA dropout by default. For larger datasets (>20 clips), consider
0.05 to prevent overfitting."""

DEFAULT_LORAPLUS_LR_RATIO: float = 4.0
"""LoRA+ learning rate ratio. The B matrix (output projection) gets
this multiplier on its learning rate relative to the A matrix.
4.0 is validated on Wan models — helps B converge faster since it
starts from zero and has more learning to do."""


# ─── Training Loop ───

DEFAULT_MAX_EPOCHS: int = 50
"""Default training duration. 50 epochs is enough for most Wan LoRA
datasets without severe overfitting."""

DEFAULT_BATCH_SIZE: int = 1
"""Video LoRA training is heavily VRAM-constrained. Batch size 1 is
the practical default — use gradient accumulation for effective larger batches."""

DEFAULT_GRADIENT_ACCUMULATION: int = 1
"""Steps between optimizer updates. Increase to simulate larger batch sizes
without additional VRAM."""

DEFAULT_SAVE_EVERY_N_EPOCHS: int = 5
"""Save a checkpoint every 5 epochs. Balances storage space against
the ability to pick the best checkpoint."""

DEFAULT_CAPTION_DROPOUT_RATE: float = 0.1
"""Probability of dropping the entire caption for a training sample.
Forces the model to rely on visual control signals (reference image)
rather than text alone. 10% is a safe default."""


# ─── Precision ───

VALID_MIXED_PRECISION: set[str] = {"bf16", "fp16", "no"}
"""Mixed precision modes for training computation.

- bf16: bfloat16. Best for modern GPUs (Ampere+). Wider dynamic range than fp16.
- fp16: float16. Works on older GPUs. Requires careful loss scaling.
- no: full fp32. Maximum precision, maximum VRAM. Rarely needed for LoRA.
"""

DEFAULT_MIXED_PRECISION: str = "bf16"
"""BFloat16 is the default for modern GPU training. It has enough dynamic
range to avoid the loss scaling issues that plague fp16."""

VALID_BASE_PRECISION: set[str] = {"fp8", "fp8_scaled", "bf16", "fp16", "fp32"}
"""Precision for the frozen base model weights.

- fp8: 8-bit float. Cuts base model VRAM roughly in half. Some quality trade-off.
- fp8_scaled: fp8 with per-tensor scaling. Better quality than plain fp8.
- bf16: full bfloat16. No quantization artifacts. Standard quality.
- fp16: float16. Similar VRAM to bf16 on modern GPUs.
- fp32: full precision. Maximum VRAM, rarely needed.
"""

DEFAULT_BASE_PRECISION: str = "fp8"
"""FP8 quantization for the frozen base model is the practical default.
Wan 14B in bf16 is ~28 GB; fp8 cuts that roughly in half, making
training possible on 24 GB GPUs."""


# ─── MoE Boundary Ratios ───

DEFAULT_BOUNDARY_RATIO_T2V: float = 0.875
"""SNR boundary for Wan 2.2 T2V expert routing. Steps with noise ratio
above this threshold use the high-noise expert; below it, the low-noise
expert. 0.875 = 875/1000 of the noise schedule."""

DEFAULT_BOUNDARY_RATIO_I2V: float = 0.900
"""SNR boundary for Wan 2.2 I2V. Slightly higher than T2V because the
reference image provides strong conditioning that shifts the effective
denoising trajectory."""


# ─── Timestep Sampling ───

VALID_TIMESTEP_SAMPLING: set[str] = {"uniform", "shift", "logit_normal", "sigmoid"}
"""How to sample noise timesteps during training.

- uniform: equal probability for all timesteps. Simple baseline.
- shift: shifted distribution favoring mid-to-high noise. Wan default.
- logit_normal: logit-normal distribution. Concentrates around the center.
- sigmoid: sigmoid-based sampling. Similar to logit_normal, different tails.
"""

DEFAULT_TIMESTEP_SAMPLING: str = "shift"
"""Wan models were trained with shifted timestep sampling. Using the same
distribution during LoRA fine-tuning maintains consistency with the
pretrained model's learned noise-level expectations."""


# ─── Flow Matching ───

DEFAULT_FLOW_SHIFT_480P: float = 3.0
"""Flow shift parameter for 480p generation. Controls the signal-to-noise
ratio curve. Higher values mean more noise early in the denoising process."""

DEFAULT_FLOW_SHIFT_720P: float = 5.0
"""Flow shift parameter for 720p generation. Higher than 480p because
larger images need more denoising steps for coherent global structure."""


# ─── Logging ───

VALID_LOG_BACKENDS: set[str] = {"console", "tensorboard", "wandb"}
"""Supported logging backends.

- console: print to terminal. Always available, no setup needed.
- tensorboard: TensorBoard log files. Good for local experiment tracking.
- wandb: Weights & Biases. Best for comparing runs across experiments.
"""


# ─── Checkpoint ───

VALID_CHECKPOINT_FORMATS: set[str] = {"safetensors", "diffusers"}
"""Checkpoint output formats.

- safetensors: single .safetensors file. ComfyUI-compatible. Default.
- diffusers: HuggingFace diffusers directory format. For diffusers pipeline loading.
"""

DEFAULT_CHECKPOINT_FORMAT: str = "safetensors"
"""Safetensors is the default because it's the most widely supported
format across inference tools (ComfyUI, A1111, diffusers)."""


# ─── Sampling ───

DEFAULT_SAMPLING_SEED: int = 42
"""Fixed seed for sample generation. Using the same seed across epochs
makes it easy to compare progress visually."""

DEFAULT_SAMPLING_STEPS: int = 30
"""Inference steps for sample generation. 30 is a good balance between
quality and speed for progress monitoring."""

DEFAULT_SAMPLING_GUIDANCE: float = 5.0
"""Guidance scale for sample generation. 5.0 is the Wan default for
balanced text adherence without over-saturation."""


# ─── Model Templates ───
# Plain dicts that the loader deep-merges as base layer before user config.
# Every key here corresponds to a field path in DimljusTrainingConfig.

MODEL_TEMPLATES: dict[str, dict] = {
    "wan21_t2v": {
        # Wan 2.1 Text-to-Video: single transformer, simplest case.
        # Good for proving the training pipeline works correctly.
        "model": {
            "family": "wan",
            "variant": "2.1_t2v",
            "is_moe": False,
            "in_channels": 16,
            "num_layers": 40,
            "flow_shift": DEFAULT_FLOW_SHIFT_480P,
        },
        "lora": {
            "rank": DEFAULT_LORA_RANK,
            "alpha": DEFAULT_LORA_ALPHA,
        },
        "optimizer": {
            "learning_rate": DEFAULT_LEARNING_RATE,
        },
        "training": {
            "max_epochs": DEFAULT_MAX_EPOCHS,
            "timestep_sampling": DEFAULT_TIMESTEP_SAMPLING,
        },
    },
    "wan22_t2v": {
        # Wan 2.2 Text-to-Video: dual MoE architecture.
        # High-noise expert handles composition/motion, low-noise handles detail.
        "model": {
            "family": "wan",
            "variant": "2.2_t2v",
            "is_moe": True,
            "in_channels": 16,
            "num_layers": 40,
            "boundary_ratio": DEFAULT_BOUNDARY_RATIO_T2V,
            "flow_shift": DEFAULT_FLOW_SHIFT_480P,
        },
        "lora": {
            "rank": DEFAULT_LORA_RANK,
            "alpha": DEFAULT_LORA_ALPHA,
        },
        "moe": {
            "enabled": True,
            "high_noise": {
                "rank": 16,
                "learning_rate": 1e-4,
                "max_epochs": 30,
            },
            "low_noise": {
                "rank": 24,
                "learning_rate": 8e-5,
                "max_epochs": 50,
            },
        },
        "optimizer": {
            "learning_rate": DEFAULT_LEARNING_RATE,
        },
        "training": {
            "max_epochs": DEFAULT_MAX_EPOCHS,
            "timestep_sampling": DEFAULT_TIMESTEP_SAMPLING,
        },
    },
    "wan22_i2v": {
        # Wan 2.2 Image-to-Video: dual MoE + reference image conditioning.
        # Reference image is VAE-encoded, concatenated with noisy latents (36 channels).
        # Higher ranks than T2V because I2V has more to learn (visual fidelity to ref).
        "model": {
            "family": "wan",
            "variant": "2.2_i2v",
            "is_moe": True,
            "in_channels": 36,
            "num_layers": 40,
            "boundary_ratio": DEFAULT_BOUNDARY_RATIO_I2V,
            "flow_shift": DEFAULT_FLOW_SHIFT_720P,
        },
        "lora": {
            "rank": DEFAULT_LORA_RANK,
            "alpha": DEFAULT_LORA_ALPHA,
        },
        "moe": {
            "enabled": True,
            "high_noise": {
                "rank": 24,
                "learning_rate": 1e-4,
                "max_epochs": 30,
            },
            "low_noise": {
                "rank": 32,
                "learning_rate": 8e-5,
                "max_epochs": 50,
            },
        },
        "optimizer": {
            "learning_rate": DEFAULT_LEARNING_RATE,
        },
        "training": {
            "max_epochs": DEFAULT_MAX_EPOCHS,
            "timestep_sampling": DEFAULT_TIMESTEP_SAMPLING,
            "caption_dropout_rate": 0.15,
        },
    },
}
"""Model template defaults, keyed by template name.

Each template defines the full set of model-specific defaults. The loader
deep-merges these as the base layer, then applies user config on top.
User values always win.

I2V has higher ranks and caption dropout because the reference image carries
more conditioning weight — the model needs more capacity to learn from it,
and higher caption dropout forces reliance on the visual signal.
"""
