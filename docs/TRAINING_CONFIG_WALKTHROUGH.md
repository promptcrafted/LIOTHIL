# Training Config Walkthrough

**Date:** 2026-02-24
**Status:** Matches approved `wan22_training_master.py` and `full_train.yaml`

This document walks through the two files that make up the Dimljus training config system and how they connect. It's a reference for understanding what exists, what decisions were made, and why.

---

## Two Files, Two Audiences

| File | Who sees it | What it does |
|------|------------|--------------|
| `examples/full_train.yaml` | **Users** | The config they fill out to run a training job |
| `dimljus/config/wan22_training_master.py` | **Nobody** (infrastructure) | Defines valid options, default values, and validation rules |

Users only ever edit the YAML. The Python file is loaded by the config system — users never open it.

A third file, `dimljus/config/training_loader.py`, is the glue: it reads the YAML, looks up variant defaults from the master file, resolves paths, and validates through Pydantic.

---

## The YAML: What Users See

The YAML is organized to follow the training flow: set up the model, configure fixed settings, configure the unified phase, configure expert overrides, then output settings.

### Section 1: Model & Data

```yaml
model:
  path: C:/Users/minta/Projects/dimljus/models/Wan2.2-T2V-14B-Diffusers
  family: wan
  variant: 2.2_t2v                         # 2.2_t2v | 2.2_i2v

data_config: ./holly/dimljus_data.yaml
```

**`variant`** is the key field. It tells the loader which architecture defaults to apply. Setting `variant: 2.2_t2v` auto-fills: `is_moe: true`, `in_channels: 16`, `num_layers: 40`, `boundary_ratio: 0.875`, `flow_shift: 3.0`. The user can override any of these if needed — the commented-out fields show what's available.

**`data_config`** points to the Dimljus data config YAML (Phase 0). The loader checks this file exists but doesn't load it — that happens at training time.

### Section 2: LoRA Structure (fixed)

```yaml
lora:
  rank: 16
  alpha: 16
  loraplus_lr_ratio: 4.0
  dropout: 0.05
  target_modules: null
  block_rank_overrides: null
```

Everything in this section is **locked for the entire training run**. Rank determines LoRA matrix dimensions — once created, the shape can't change. Alpha sets the scaling factor (alpha/rank = effective strength).

**Targeting** uses two levels:
- **Component level:** `"ffn"`, `"self_attn"`, `"cross_attn"` — trains all projections in that component
- **Projection level:** `"cross_attn.to_v"`, `"ffn.up_proj"` — trains one specific projection

`null` = all standard targets (all attention + FFN). The commented guide in the YAML explains the rules.

**Block rank overrides** let you give different LoRA capacity to different transformer blocks. Early blocks (0-11) handle composition, late blocks (30-39) handle detail — you might want more capacity where it matters most.

### Section 3: Optimizer

```yaml
optimizer:
  # Fixed for entire run
  type: adamw8bit
  betas: [0.9, 0.999]
  eps: 1e-8
  max_grad_norm: 1.0
  optimizer_args: {}

  # Unified starting values — overridable per expert after fork
  learning_rate: 5e-5
  weight_decay: 0.01
```

This section is split into **fixed** and **overridable** zones.

Fixed settings (optimizer type, betas, eps, gradient clipping) stay constant for the entire run. You pick your optimizer once.

Learning rate and weight decay are **starting values** — they apply during the unified phase and each expert inherits them after fork, unless overridden per-expert.

**Why 5e-5?** Community recommendations (2e-4) are too aggressive for Wan 2.2, especially for the low-noise expert which overfits rapidly at high LR. 5e-5 is conservative. Each expert overrides as needed.

### Section 4: Scheduler

```yaml
scheduler:
  # Fixed for entire run
  warmup_steps: 0

  # Unified starting values — overridable per expert after fork
  type: cosine_with_min_lr
  min_lr: null
  min_lr_ratio: 0.01
```

Same fixed/overridable split. Warmup runs once at training start (fixed). Scheduler type and min LR ratio are overridable per expert — high-noise might want faster decay, low-noise might want constant LR.

### Section 5: Training

```yaml
training:
  # Fixed for entire run
  mixed_precision: bf16
  base_model_precision: bf16
  timestep_sampling: shift
  gradient_checkpointing: true
  seed: 42

  # Unified phase — starting values, overridable per expert after fork
  unified_epochs: 10
  unified_targets: null
  unified_block_targets: null
  batch_size: 1
  gradient_accumulation_steps: 1
  caption_dropout_rate: 0.10
```

Fixed settings: precision (`bf16` for both training and frozen base model — no quantization shortcuts), timestep sampling (`shift` matches Wan's pretraining), gradient checkpointing (always on for video).

Unified phase settings: `unified_epochs` controls how long both experts share a single LoRA before forking. `unified_targets` and `unified_block_targets` can narrow what gets trained during the shared phase.

**Why bf16 for base model?** Quality first. fp8 is available if VRAM-constrained but introduces quantization artifacts. The master file documents this tradeoff.

### Section 6: MoE Expert Fork

```yaml
moe:
  enabled: true
  fork_enabled: true
  boundary_ratio: null

  high_noise:
    learning_rate: 1e-4
    max_epochs: 30

  low_noise:
    learning_rate: 8e-5
    max_epochs: 50
```

**`enabled`** — this model has two noise-level experts. Both are active during training regardless of mode. Even in unified-only mode (fork_enabled: false), both experts route during the forward pass.

**`fork_enabled`** — the master switch for fork-and-specialize. Three training modes:

| Mode | Settings | What happens |
|------|----------|-------------|
| Fork-and-specialize | `fork_enabled: true` + `unified_epochs > 0` | Unified LoRA warmup → fork into per-expert copies → each trains independently |
| Unified only | `fork_enabled: false` | One LoRA on both experts, no forking |
| Expert from scratch | `fork_enabled: true` + `unified_epochs: 0` | Skip unified, go straight to per-expert |

**Per-expert overrides** — `null` means inherit from the unified/optimizer/scheduler defaults. Only set what you want to change. The full list of overridable fields per expert:

- `learning_rate`, `max_epochs`, `dropout`
- `fork_targets`, `block_targets` (narrow what gets trained per expert)
- `resume_from` (start from an existing LoRA file)
- `batch_size`, `gradient_accumulation_steps`, `caption_dropout_rate`
- `weight_decay`, `min_lr_ratio`
- `optimizer_type`, `scheduler_type`

### Section 7: Checkpoints

```yaml
save:
  save_every_n_epochs: 5
  output_dir: ./output/holly_i2v
  name: holly_lora
  format: safetensors
```

During fork-and-specialize, checkpoints are organized automatically:

```
{output_dir}/
  unified/
    {name}_epoch005.safetensors
    {name}_epoch010.safetensors
  high_noise/
    {name}_high_epoch015.safetensors
  low_noise/
    {name}_low_epoch020.safetensors
```

### Section 8: Logging

```yaml
logging:
  backends: [console, wandb]
  wandb_project: dimljus-training
  wandb_entity: null
  wandb_run_name: holly_i2v_r16
```

W&B requires `wandb login` or `WANDB_API_KEY` env var before use. The validator catches missing `wandb_project` when wandb is in backends.

### Section 9: Sampling

```yaml
sampling:
  enabled: true
  every_n_epochs: 5
  prompts:
    - "Holly Golightly walks elegantly through a sunlit garden"
  neg: "blurry, low quality, distorted"
  seed: 42
  walk_seed: true
  sample_steps: 30
  guidance_scale: 5.0
  sample_dir: null
```

Off by default — inference passes are expensive. `walk_seed` increments the seed per prompt for variety while keeping each individual prompt reproducible across epochs.

---

## The Master File: What's Inside

`wan22_training_master.py` has three parts.

### Part 1: Valid Options (lines 23-155)

Vocabulary — what names are legal in config fields. Seven sets:

| Set | Count | Examples |
|-----|-------|---------|
| `VALID_OPTIMIZERS` | 7 | adamw, adamw8bit, adafactor, came, prodigy, ademamix, schedule_free_adamw |
| `VALID_SCHEDULERS` | 11 | constant, cosine, cosine_with_min_lr, polynomial, rex, warmup_stable_decay... |
| `VALID_MIXED_PRECISION` | 3 | bf16, fp16, no |
| `VALID_BASE_PRECISION` | 5 | fp8, fp8_scaled, bf16, fp16, fp32 |
| `VALID_TIMESTEP_SAMPLING` | 4 | uniform, shift, logit_normal, sigmoid |
| `VALID_LOG_BACKENDS` | 3 | console, tensorboard, wandb |
| `VALID_FORK_TARGETS` | 12 | Component: ffn, self_attn, cross_attn. Projection: cross_attn.to_q, ffn.up_proj... |

Each has a docstring explaining what every option does and when to use it.

### Part 2: Default Values (lines 158-563)

All constants with `T2V_` or `I2V_` prefixes. Organized into zones:

**Architecture** — model family, variant, channels, layers, boundary ratio, flow shift. These get applied by the loader when a user sets `variant: 2.2_t2v`.

**Training Strategy** — `T2V_FORK_ENABLED = True`. The master switch. One constant, three training modes via YAML settings.

**Fixed Settings** — rank (16), alpha (16), loraplus (4.0), optimizer type (adamw8bit), betas, eps, grad norm, warmup (0), mixed precision (bf16), base model precision (bf16), timestep sampling (shift). These don't change between phases.

**Unified Foundation** — unified_epochs (10), learning_rate (5e-5), weight_decay (0.01), scheduler (cosine_with_min_lr), min_lr_ratio (0.01), lora_dropout (0.0), batch_size (1), gradient_accumulation (1), caption_dropout (0.10). Starting values that experts can override.

**Expert Overrides** — Per-expert fields for high-noise and low-noise. All `None` by default = inherit from unified. `max_epochs` has a value (50 for both).

**I2V Differences** — Only the values that differ from T2V:
- `I2V_IN_CHANNELS = 36` (16 noise + 20 reference image)
- `I2V_BOUNDARY_RATIO = 0.900` (vs 0.875)
- `I2V_UNIFIED_EPOCHS = 15` (longer — more shared signal)
- `I2V_CAPTION_DROPOUT_RATE = 0.15` (higher — reference image carries conditioning)

**Variant Defaults Map** — `VARIANT_DEFAULTS` dict maps `"2.2_t2v"` and `"2.2_i2v"` to their override dicts. The loader deep-merges these under the user's YAML.

### Part 3: Pydantic Schema (lines 568-end)

11 models that define the YAML structure and validation:

| Model | Key fields |
|-------|-----------|
| `ModelConfig` | path (required), family, variant, is_moe, in_channels, boundary_ratio, flow_shift |
| `LoraConfig` | rank, alpha, dropout, loraplus_lr_ratio, target_modules, block_rank_overrides, use_mua_init |
| `OptimizerConfig` | type, learning_rate, weight_decay, betas, eps, max_grad_norm |
| `SchedulerConfig` | type, warmup_steps, min_lr, min_lr_ratio, rex_alpha, rex_beta |
| `MoeExpertOverrides` | All nullable: learning_rate, dropout, max_epochs, fork_targets, block_targets, resume_from, batch_size, gradient_accumulation_steps, caption_dropout_rate, weight_decay, min_lr_ratio, optimizer_type, scheduler_type |
| `MoeConfig` | enabled, fork_enabled, high_noise, low_noise, boundary_ratio |
| `TrainingLoopConfig` | unified_epochs, unified_targets, unified_block_targets, batch_size, gradient_accumulation_steps, mixed_precision, base_model_precision, caption_dropout_rate, timestep_sampling, seed, resume_from |
| `SaveConfig` | output_dir, name, save_every_n_epochs, format |
| `LoggingConfig` | backends, wandb_project, wandb_entity, wandb_run_name |
| `SamplingConfig` | enabled, every_n_epochs, prompts, neg, seed, walk_seed, sample_steps, guidance_scale, sample_dir |
| `DimljusTrainingConfig` | Root — assembles all of the above |

**Root validators** (cross-field checks):
1. `check_moe_consistency` — error if MoE enabled on non-MoE model
2. `check_prodigy_lr` — error if Prodigy with lr != 1.0
3. `check_wandb_project` — error if wandb backend but no project name
4. `check_mua_alpha` — auto-set alpha=rank when muA enabled
5. `check_fork_without_moe` — error if fork_enabled but MoE disabled
6. `warn_aggressive_low_noise` — soft warning if low-noise LR > 2e-4

---

## The Loader: How They Connect

`training_loader.py` runs this sequence:

```
1. Find YAML (file path or directory with dimljus_train.yaml)
2. Load YAML → dict
3. Look up model.variant in VARIANT_DEFAULTS
4. Deep-merge: variant defaults as base, user YAML on top (user wins)
5. Auto-enable moe.enabled if is_moe is true
6. Resolve all relative paths (data_config, model.path, save.output_dir,
   training.resume_from, expert resume_from paths, sampling.sample_dir)
7. Validate through Pydantic → DimljusTrainingConfig
8. Check data_config file exists
```

**Deep merge rules:**
- Scalars: user wins
- Dicts: recurse (nested merge)
- Lists: user replaces entirely (no list merging)

This means a user who sets `optimizer.learning_rate: 1e-4` overrides the default `5e-5` without affecting any other optimizer field.

---

## Key Decisions Made During Review

These were decided during the Phase 5 file-by-file review (2026-02-24):

| Decision | Rationale |
|----------|-----------|
| `template` → `variant` | The YAML is the config users fill out. "template" implied the Python file was user-facing. `variant` describes the model. |
| `exclude_modules` removed | Redundant with `target_modules` — both are ways to say "train these, not those" |
| `freeze_shared_after_fork` removed | Research flag, not validated, not v1 |
| `fork_criterion` removed | `loss_plateau` detection not built, not v1 |
| Per-expert rank removed | Rank = matrix dimensions, locked at creation, can't change after fork |
| `max_epochs` removed from TrainingLoopConfig | Redundant — total duration = unified_epochs + per-expert max_epochs |
| LR: 2e-4 → 5e-5 | Community recommendations too aggressive for Wan 2.2, especially low-noise expert |
| Base precision: fp8 → bf16 | Quality first, no quantization shortcuts. fp8 available if VRAM-constrained. |
| Rank: 32 → 16 | Updated to match production settings |
| Alpha matches rank | 1.0x neutral scaling (was 0.5x with alpha=16, rank=32) |
| One master .py, multiple YAMLs | T2V and I2V share the same schema with different default values |
| Fixed/unified/expert zones | Each YAML section splits into fixed fields first, then overridable unified fields |
| `wandb_entity` added | Needed for functional W&B integration |

---

## Current Defaults: Side-by-Side with Production

Dimljus defaults vs. Minta's current musubi production settings for Wan 2.2 T2V high-noise expert:

| Setting | Dimljus default | Musubi production |
|---------|----------------|-------------------|
| LoRA rank | 16 | 16 |
| LoRA alpha | 16 | 16 |
| Learning rate | 5e-5 | 5e-5 |
| Optimizer | adamw8bit (wd=0.01) | adamw8bit (wd=0.01) |
| Scheduler | cosine_with_min_lr | polynomial (power=2) |
| Epochs | 50 | 50 |
| Save interval | 5 epochs | 5 epochs |
| Batch size | 1 | 1 |
| Gradient accumulation | 1 | 1 |
| Mixed precision | bf16 | fp16 |
| Base model precision | bf16 | fp8_scaled |
| Seed | 42 | 42 |
| Flow shift | 3.0 | 3.0 |
| Timestep boundary | 0.875 | 875 |

Differences to watch: scheduler type, mixed precision, base model precision. The scheduler difference (cosine vs polynomial) may warrant changing the default or making it a per-variant recommendation.
