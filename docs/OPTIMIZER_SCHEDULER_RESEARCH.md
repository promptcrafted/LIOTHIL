# Optimizer & Scheduler Research for Dimljus Training Config

Research conducted 2026-02-23 for Phase 5 (Training Config Schema) design decisions.
Covers optimizers, LR schedulers, and LoRA-specific techniques relevant to Wan 2.1/2.2 video LoRA training.

---

## Table of Contents

1. [CAME Optimizer](#1-came-optimizer)
2. [Rex Scheduler](#2-rex-scheduler)
3. [LoRA+](#3-lora)
4. [Prodigy Optimizer](#4-prodigy-optimizer)
5. [AdEMAMix Optimizer](#5-ademamix-optimizer)
6. [Schedule-Free Optimizers](#6-schedule-free-optimizers)
7. [Community Recommendations for Wan LoRA Training](#7-community-recommendations-for-wan-lora-training)
8. [Comparative Summary & Dimljus Defaults](#8-comparative-summary--dimljus-defaults)

---

## 1. CAME Optimizer

**Paper:** "CAME: Confidence-guided Adaptive Memory Efficient Optimization" (Luo et al., ACL 2023, arXiv:2307.02047)

### What It Is

CAME is a memory-efficient adaptive optimizer that achieves Adam-level convergence quality with Adafactor-level memory usage. The key innovation is a **confidence-guided strategy** that stabilizes the factorized second-moment estimation used by memory-efficient optimizers like Adafactor.

**Core mechanism:** Instead of storing the full second-moment matrix V (like Adam), CAME factorizes it into row and column factors (like Adafactor), but adds a **confidence matrix** that weights the update based on how reliable the factored approximation is for each parameter. When the factored estimate is unreliable (high disagreement between row/column factors), the confidence weight reduces the update magnitude, preventing the instability that plagues Adafactor.

### Default Hyperparameters

```python
from came_pytorch import CAME

optimizer = CAME(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999, 0.9999),   # THREE betas (not two like Adam)
    eps=(1e-30, 1e-16)             # TWO eps values (not one like Adam)
)
```

- **betas**: Tuple of three values. `beta1` (0.9) = gradient EMA, `beta2` (0.999) = row/column factor EMA, `beta3` (0.9999) = confidence EMA. The third beta controls how quickly the confidence metric adapts.
- **eps**: Tuple of two values. `eps1` (1e-30) = numerical stability for row/column factorization, `eps2` (1e-16) = numerical stability for the update.
- **weight_decay**: Default 1e-2, same as AdamW convention.

### Memory Usage vs AdamW

| Optimizer | States per Parameter | Memory for 14B Model (fp32 states) |
|-----------|---------------------|-----------------------------------|
| AdamW     | 2 full tensors (m, v) | ~8 bytes/param => very large |
| Adafactor | 2 vectors per matrix (row + col factors) | ~4 bytes/param equivalent |
| CAME      | Same as Adafactor + confidence row/col factors | ~4-5 bytes/param equivalent |
| Adam 8-bit (bitsandbytes) | 2 tensors quantized to 8-bit | ~2 bytes/param |

For LoRA training specifically, the optimizer state overhead is proportional to LoRA parameters only (not the full model), so the memory savings of CAME vs AdamW are modest in absolute terms for typical LoRA ranks (32-128). The savings become more significant for high-rank LoRA, full fine-tuning, or VRAM-constrained setups.

### Community Experience

- CAME is available in musubi-tuner via `--optimizer_type came_pytorch` (requires `pip install came-pytorch`).
- Mentioned in the Derrian Distro trainer as `LoraEasyCustomOptimizer.came.CAME`.
- The Civitai "Opinionated Guide to All Lora Training (2025 Update)" notes: "there is the CAME optimizer... I have yet personally to test this, but there are plenty of my peers who use this and are totally good with this."
- Community adoption for diffusion LoRA training is **moderate** -- less common than AdamW8bit or Adafactor, but growing. Not widely tested specifically on Wan models yet.

### Known Issues / Gotchas

- Three betas (not two) -- config schemas must handle this.
- Two eps values (not one) -- unusual, easy to misconfigure.
- The PyPI package is `came-pytorch`, not built into PyTorch or bitsandbytes.
- No 8-bit quantized version exists, so memory savings come only from the factorization, not from quantization.
- Limited community validation for video LoRA specifically.

### Dimljus Relevance: MEDIUM

CAME offers a principled middle ground between "full Adam quality" and "Adafactor memory savings." Relevant for users who want Adam-quality convergence but are VRAM-constrained. Should be supported but not the default.

---

## 2. Rex Scheduler

**Paper:** "REX: Revisiting Budgeted Training with an Improved Schedule" (Chen, Wolfe, Kyrillidis, MLSys 2022, arXiv:2107.04197)

### What It Is

Rex (Reflected Exponential) is a learning rate schedule that uses an exponential decay profile reflected around the midpoint of training. The key insight: an exponential decay is good at the start (high LR for exploration) but decays too fast, while the reverse-exponential is good at the end. Rex combines both by reflecting the exponential curve.

**Formula (simplified):** The LR at step t follows a curve that starts near max_lr, maintains a relatively high LR through the middle of training, then drops sharply near the end. It looks like a slightly "squashed" cosine schedule but with different mathematical properties.

### Parameters

As implemented in musubi-tuner (based on IvanVassi's implementation):

- **rex_alpha**: Controls the exponential decay rate. Default **0.1** in musubi-tuner (paper suggests 0.5).
- **rex_beta**: Controls the balance between the two exponential components. Default **0.9** in musubi-tuner (paper suggests 0.5).
- **lr_warmup_steps**: Optional warmup period (default 0).
- **lr_scheduler_min_lr_ratio**: Minimum LR as fraction of max LR (default 0.01).

From musubi-tuner docs:
> "It has two parameters, rex_alpha and rex_beta, with default values of 0.1 and 0.9, respectively. These parameters are based on the defaults in IvanVassi's repository. The values proposed in the paper are 0.5 and 0.5."

### Comparison to Other Schedules

| Schedule | Behavior | Best For |
|----------|----------|----------|
| Constant | Flat LR throughout | Simple, Prodigy pairing |
| Cosine | Smooth decrease from max to ~0 | General purpose, widely used |
| Linear | Linear decrease from max to 0 | Budget-constrained training |
| Rex | High LR maintained longer, sharp drop at end | Budget-aware, any duration |
| Cosine with min LR | Cosine but floors at min_lr | Preventing collapse at end |

Rex outperforms linear and cosine in **low-budget regimes** (few epochs/steps) while matching or exceeding them in high-budget regimes. This is particularly relevant for LoRA training where you often don't know the optimal step count in advance.

### Community Experience

- Available in musubi-tuner via `--lr_scheduler rex`.
- Available in Axolotl (documented in their scheduler API).
- Community adoption is **low but growing**. Most Wan LoRA trainers still use constant, cosine, or cosine_with_min_lr.
- One Civitai guide recommends "1e-4 + cosine with min lr 0.01" as a versatile starting point, not Rex specifically.

### Dimljus Relevance: MEDIUM-HIGH

Rex is architecturally interesting for Dimljus because of the differential MoE training strategy. High-noise experts converge faster (fewer epochs needed), low-noise experts need more. A schedule that works well across budget regimes could simplify the per-expert scheduler configuration. Worth supporting and testing.

---

## 3. LoRA+

**Paper:** "LoRA+: Efficient Low Rank Adaptation of Large Models" (Hayou, Ghosh, Yu, arXiv:2402.12354, Feb 2024)

### What It Is

LoRA+ observes that standard LoRA trains both adapter matrices A (down-projection) and B (up-projection) with the **same learning rate**, but this is suboptimal for large-width models. Matrix B (the "up" direction, initialized to zero) should learn faster than matrix A (the "down" direction, initialized randomly).

**The fix is simple:** Multiply the LR for matrix B by a ratio (typically 2-16x). That's it. No new optimizer, no architectural changes.

### Implementation

In musubi-tuner (and kohya sd-scripts), this is a **network_arg**, not an optimizer setting:

```
--network_args "loraplus_lr_ratio=4"
```

The implementation creates **separate parameter groups** for LoRA-A and LoRA-B weights, with the B group getting `base_lr * loraplus_lr_ratio` as its learning rate. Any optimizer can be used with LoRA+.

### Typical Values

| Setting | Context |
|---------|---------|
| `loraplus_lr_ratio=2` | Conservative, good starting point |
| `loraplus_lr_ratio=4` | **Recommended by musubi-tuner docs and community** |
| `loraplus_lr_ratio=16` | Paper's recommendation, but community finds it too aggressive |

From musubi-tuner advanced_config.md:
> "The original paper recommends 16, but it may need adjustment. Starting around 4 seems to work well."

From the "WAN training - Rules of the trade" discussion:
> "Wan, like Hunyuan also seems to benefit from LoraPlus (loraplus_lr_ratio=X where X is the multiplier for the LR on lora_b blocks, 2 or 4 seems good for Wan)"

### Important Notes

- LoRA+ is **NOT compatible with LoHa or LoKr** -- it is specific to standard LoRA's A/B matrix structure.
- Works with any optimizer (AdamW, Adafactor, CAME, Prodigy, etc.).
- The DeepWiki docs note: "Start with loraplus_lr_ratio=4. Original paper recommends 16, but lower values often work better."
- A Feb 2026 paper (arXiv:2602.04998) found that "once learning rates are properly tuned, all [LoRA] methods achieve similar peak performance (within 1-2%)" -- suggesting LoRA+ mainly helps when you haven't perfectly tuned the base LR.

### Dimljus Relevance: HIGH

This is a per-network configuration, not a per-optimizer configuration. Dimljus should support `loraplus_lr_ratio` as a LoRA network argument with a default of 4. It's widely validated on Wan and essentially free (no extra compute or memory).

**Config location:** Should live in the LoRA network config, not the optimizer config. Something like:

```yaml
lora:
  rank: 32
  alpha: 16
  loraplus_lr_ratio: 4  # multiplier for B/up matrix LR
```

---

## 4. Prodigy Optimizer

**Paper:** "Prodigy: An Expeditiously Adaptive Parameter-Free Learner" (Mishchenko & Defazio, arXiv:2306.06101, ICML 2023 workshop)

### What It Is

Prodigy is an **adaptive learning rate optimizer** that estimates the optimal learning rate automatically during training. You set `lr=1` and Prodigy figures out the actual scale. It is an improvement on D-Adaptation (same author, Aaron Defazio) with faster convergence of the LR estimate.

**Core idea:** Prodigy estimates the "distance to solution" D, which determines the optimal LR. It does this by tracking gradient statistics and progressively refining its estimate. The user doesn't need to tune the learning rate.

### Default Hyperparameters

```python
from prodigyopt import Prodigy

optimizer = Prodigy(
    model.parameters(),
    lr=1.0,               # ALWAYS set to 1 (Prodigy scales internally)
    betas=(0.9, 0.99),    # Note: 0.99, not 0.999 like Adam
    weight_decay=0.01,    # 0 to 0.1
    d_coef=1.0,           # multiplier on estimated LR (>1 = more aggressive)
    use_bias_correction=True,
    safeguard_warmup=True,
    decouple=True
)
```

### Typical Settings for Diffusion LoRA

From community consensus:
```
--optimizer_type prodigyopt.Prodigy
--learning_rate 1
--optimizer_args decouple=True weight_decay=0.01 d_coef=0.8 use_bias_correction=True safeguard_warmup=True betas=0.9,0.99
```

Key tuning knobs:
- **d_coef**: The de facto "learning rate" control. Values 0.5-2.0. Lower = more conservative. Community recommends **0.8** for LoRA training.
- **weight_decay**: 0.01 is standard. Higher (0.1-0.9) reduces overfitting at cost of likeness.
- **betas**: (0.9, 0.99) -- note beta2 is lower than Adam's default 0.999.

### Comparison to AdamW

| Aspect | Prodigy | AdamW |
|--------|---------|-------|
| LR tuning needed | No (set lr=1) | Yes (critical) |
| Convergence speed | Fast initial convergence | Depends on LR |
| Overfitting risk | Higher (aggressive LR) | Controllable |
| Memory | Same as Adam (2 states + D estimate) | Same |
| Scheduler | Constant or cosine (not restarts) | Any |
| Batch size | Best with batch=1 (per some reports) | Any |

### Community Experience with Diffusion LoRA

**Positives:**
- "Prodigy is blazing fast!" -- r/StableDiffusion
- "Simplifies hyperparameter tuning by adapting learning rates automatically"
- "Effective for fine-tuning tasks like DreamBooth LoRA training"
- Works in musubi-tuner: `--optimizer_type prodigyopt.Prodigy --learning_rate 1`

**Negatives:**
- "But it overtrains like a mf!" -- common complaint
- "I am skeptical of Prodigy. Given it adapts to try and minimize loss, I don't believe it can really pick a proper learning rate for you. It's essentially just a game of suddenly tuning different hyperparameters (d_coef and weight_decay)."
- "Limited research and practical implementation may hinder its adoption" -- Shakker AI guide
- Reports of it performing poorly with larger batch sizes

### Prodigy-Plus-Schedule-Free

A community fork (`prodigy-plus-schedule-free` on PyPI) combines Prodigy with Schedule-Free learning, StableAdamW, and low-rank second-moment factorization (like Adafactor). Features:
- Schedule-Free can be disabled (`use_schedulefree=False`)
- StableAdamW gradient clipping built in
- Per-parameter-group D estimation (`split_groups=True`)
- Low-rank approximations for second moments (memory savings)
- Supported in OneTrainer

### Dimljus Relevance: MEDIUM

Prodigy is appealing for "just works" simplicity, but the overfitting tendency and the fact that it shifts tuning to d_coef/weight_decay rather than eliminating tuning makes it less compelling for an opinionated trainer. Should be supported, but not recommended as default for Wan video LoRA where careful LR control matters (especially with differential MoE).

**Gotcha for differential MoE:** Prodigy estimates a single D value across all parameters by default. For differential training (different hyperparams per expert), each expert would need its own parameter group with separate D estimation. The `prodigy-plus-schedule-free` fork supports this via `split_groups=True`.

---

## 5. AdEMAMix Optimizer

**Paper:** "The AdEMAMix Optimizer: Better, Faster, Older" (Pagliardini, Ablin, Grangier, arXiv:2409.03137, Sep 2024, Apple)

### What It Is

AdEMAMix modifies Adam by using a **mixture of two Exponential Moving Averages (EMAs)** for the first moment, instead of just one. The insight: standard Adam's single EMA (with beta1=0.9) exponentially forgets older gradients too quickly. Some older gradients remain useful for tens of thousands of steps.

**Core mechanism:** Instead of one momentum term with beta1=0.9, AdEMAMix maintains:
- A **fast EMA** (beta1=0.9) -- same as Adam, captures recent gradient direction
- A **slow EMA** (beta3=0.9999) -- captures long-term gradient trends
- An **alpha parameter** (default 8.0) that controls the mixing weight

The update combines both EMAs: `m = m_fast + alpha * m_slow`

### Default Hyperparameters

```python
from ademamix import AdEMAMix

optimizer = AdEMAMix(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999, 0.9999),  # (beta1_fast, beta2, beta3_slow)
    alpha=8.0,                     # mixing weight for slow EMA
    weight_decay=0.1,
    # Warmup schedules for alpha and beta3:
    T_alpha=num_iterations,        # alpha warmup duration
    T_beta3=num_iterations,        # beta3 warmup duration
)
```

### Key Results

The headline claim: "a 1.3B parameter AdEMAMix LLM trained on 101B tokens performs comparably to an AdamW model trained on 197B tokens (+95%)." This means AdEMAMix reaches the same quality with roughly half the training tokens.

Additionally: "our method significantly slows down model forgetting during training."

### Memory Overhead

AdEMAMix stores **one additional EMA** (the slow momentum buffer) per parameter compared to Adam. So:
- Adam: 2 states per param (m, v)
- AdEMAMix: 3 states per param (m_fast, m_slow, v)
- ~50% more optimizer state memory than Adam

For LoRA training, a Reddit user noted: "with LoRA fine-tuning, the optimizer states don't use much memory, so this new optimizer could be good for that."

### A Simplified Variant

A Feb 2025 paper (arXiv:2502.02431, "Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD") proposes **Simplified-AdEMAMix** which uses a single momentum term while matching AdEMAMix performance. This eliminates the extra state.

### Community Experience with LoRA Training

- **Very limited community adoption for diffusion model LoRA.** No reports found of AdEMAMix being used for Wan training.
- Not available as a built-in option in musubi-tuner, ai-toolkit, or OneTrainer (would need custom import via `pytorch_optimizer` package).
- The r/LocalLLaMA subreddit discussion focused on LLM pretraining, not LoRA.
- Theoretical advantage is greatest for **long training runs** -- the slow EMA needs thousands of steps to build up useful signal. LoRA training runs (typically 500-5000 steps) may be too short to benefit.

### Dimljus Relevance: LOW

Interesting research but not practical for Dimljus Phase 5. The training runs are too short for the slow EMA to provide benefit, no community validation exists for diffusion LoRA, and the extra memory state is a cost with uncertain payoff. File under "watch for future investigation."

---

## 6. Schedule-Free Optimizers

**Paper:** "The Road Less Scheduled" (Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky, arXiv:2405.15682, May 2024, Meta)

### What It Is

Schedule-Free learning eliminates the need for a learning rate schedule entirely. Instead of decaying the LR on a predetermined curve (cosine, linear, etc.), the optimizer uses a **combination of interpolation and iterate averaging** to achieve the same effect automatically.

**Key properties:**
- **No schedule hyperparameters** (no warmup steps, no decay steps, no T_max)
- Same hyperparameters as standard Adam/SGD: just lr, betas, weight_decay
- Matches or outperforms cosine annealing across tasks
- Requires calling `optimizer.eval()` before evaluation and `optimizer.train()` before training (maintains two sets of parameters internally)

### Implementation

Available via Meta's official implementation:
```bash
pip install schedulefree
```

```python
import schedulefree

optimizer = schedulefree.AdamWScheduleFree(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# Before evaluation:
optimizer.eval()
# Before training:
optimizer.train()
```

### Community Experience

- One Reddit user on r/StableDiffusion: **"The Schedule Free Adam Optimizer seems to produce the best results if you have the hardware to run it."** (OneTrainer SDXL context)
- Available in OneTrainer's optimizer list.
- The `prodigy-plus-schedule-free` package combines Prodigy's adaptive LR with Schedule-Free's iterate averaging.
- **NOT natively available in musubi-tuner** as a built-in option, though can likely be used via custom optimizer specification.
- Not widely tested on Wan models specifically.

### Gotcha: eval()/train() Mode Switching

Schedule-Free requires explicit mode switching between training and evaluation. The model parameters during training are NOT the same as the averaged parameters used for inference. This means:
- Sample generation during training needs `optimizer.eval()` first
- This complicates mid-training sampling workflows
- Some training frameworks may not handle this correctly

### Dimljus Relevance: LOW-MEDIUM

Interesting conceptual fit (eliminates scheduler tuning), but the eval/train mode switching is a complication, and lack of Wan-specific validation is a concern. Support as an advanced option, not default.

---

## 7. Community Recommendations for Wan LoRA Training

This section synthesizes actual configurations used by the Wan LoRA training community across musubi-tuner, ai-toolkit, and diffusion-pipe.

### The "Known Good" Baseline (musubi-tuner)

Based on analysis of 20+ publicly shared training configs from GitHub discussions, Civitai guides, and Reddit:

```
--optimizer_type adamw8bit
--learning_rate 2e-4
--network_dim 32
--network_alpha 16 or 32
--network_args "loraplus_lr_ratio=4"
--timestep_sampling shift
--discrete_flow_shift 3.0 (Wan 2.1) or 5.0 (Wan 2.2)
--gradient_checkpointing
--fp8_base
--mixed_precision bf16 or fp16
```

### Optimizer Popularity (from config analysis)

| Optimizer | Frequency | Notes |
|-----------|-----------|-------|
| **adamw8bit** | ~60% | Dominant choice. Reliable, well-understood |
| **adamw** (fp32) | ~20% | Used when VRAM allows |
| **adafactor** | ~10% | Memory-constrained setups, fused backward pass |
| **adamw_optimi** | ~5% | Taz's anime guide default |
| **prodigy** | ~3% | Niche, requires specific setup |
| **came_pytorch** | ~2% | Growing but still rare |

### Learning Rate Ranges

| Model | Recommended LR | Source |
|-------|---------------|--------|
| Wan 2.1 T2V 14B | 2e-4 to 3e-4 | Multiple guides |
| Wan 2.1 I2V 14B | 2e-4 | Community consensus |
| Wan 2.2 T2V high noise | 3e-4 | Civitai workflow guide |
| Wan 2.2 T2V low noise | 3e-4 | Same source (some use 1.6e-4) |
| Wan 2.2 I2V high noise | 2e-4 | musubi-tuner discussions |
| Wan 2.2 I2V low noise | 1.6e-4 to 2e-4 | Various configs |
| General range | **1e-4 to 3e-4** | Safe range for all Wan variants |

### Scheduler Popularity

| Scheduler | Usage | Notes |
|-----------|-------|-------|
| **constant** | ~40% | Simple, works well with adamw8bit |
| **cosine** | ~25% | Smooth decay, common with warmup |
| **cosine_with_min_lr** | ~20% | Recommended: "1e-4 + cosine with min lr 0.01" |
| **rex** | ~5% | Available in musubi-tuner, gaining interest |
| **linear** | ~5% | Budget-aware |
| **polynomial** | ~5% | Occasionally used |

### LoRA Rank Settings

| Setting | Common Values | Notes |
|---------|--------------|-------|
| network_dim | 16, 32, **64** | 32 is most common, 64 for quality |
| network_alpha | dim/2 or dim | 16 with dim=32, or 32 with dim=32 |
| loraplus_lr_ratio | 2 or **4** | 4 is community standard |

### Key Community Insights

1. **"Wan, like Hunyuan, seems to benefit from LoRA+"** -- early community finding that has become standard practice.

2. **Alpha scaling matters.** "Disabling it tends to make training more unforgiving but if you get it right the result might be better. Note that alpha scaling is supported by kohya's projects but not diffusion-pipe."

3. **fp8_scaled helps.** For Wan 2.2, `--fp8_scaled` in combination with `--fp8_base` uses a scaling algorithm that improves quality with fp8 models.

4. **Fused backward pass** with Adafactor supports stochastic rounding in musubi-tuner, which is important for bf16 LoRA weights. Not all optimizers support this.

5. **Warmup is optional** for Wan LoRA. Most configs use 0 warmup steps, though some cosine configs use 100 warmup steps.

6. **Dropout** is used by some: "dropout: 0.05 under network -- necessary for datasets over 20 clips."

### ai-toolkit Defaults (ostris)

The ai-toolkit Wan 2.1 14B example config uses:
- Optimizer: AdamW8bit (implied by default)
- LR: 1e-4 (slightly lower than musubi-tuner community)
- Rank: 32
- Common LR range: 0.0001 - 0.0002

---

## 8. Comparative Summary & Dimljus Defaults

### Optimizer Recommendation Matrix

| Optimizer | Stability | LR Tuning | Memory | Wan Validated | Recommended For |
|-----------|-----------|-----------|--------|---------------|-----------------|
| **AdamW8bit** | Excellent | Manual | Low | Yes (extensive) | **Default choice** |
| **AdamW** | Excellent | Manual | Medium | Yes | When VRAM allows |
| **Adafactor** | Good | Self-tuning option | Lowest | Yes | VRAM-constrained |
| **CAME** | Good | Manual | Low-Medium | Limited | Memory-conscious quality |
| **Prodigy** | Moderate | Auto (d_coef) | Medium | Limited | Quick experiments |
| **AdEMAMix** | Unknown | Manual | Higher | None | Not recommended yet |
| **Schedule-Free AdamW** | Good | Manual (no sched) | Medium | None | Advanced users |

### Scheduler Recommendation Matrix

| Scheduler | Simplicity | Budget-Robust | Wan Validated | Recommended For |
|-----------|-----------|---------------|---------------|-----------------|
| **constant** | Simplest | Yes | Yes (extensive) | Default for Prodigy, simple setups |
| **cosine_with_min_lr** | Simple | Moderate | Yes | **Default choice** |
| **cosine** | Simple | Moderate | Yes | General purpose |
| **rex** | Moderate | Excellent | Limited | Budget-uncertain training |
| **linear** | Simple | Good | Limited | Short runs |

### Proposed Dimljus Defaults

Based on community consensus and the specific needs of Wan video LoRA training:

```yaml
optimizer:
  type: adamw8bit           # most validated, good VRAM/quality balance
  learning_rate: 2e-4       # community sweet spot for Wan 14B
  weight_decay: 0.01        # standard
  betas: [0.9, 0.999]       # Adam defaults
  eps: 1e-8                 # Adam default

scheduler:
  type: cosine_with_min_lr  # smooth decay, prevents LR collapse
  warmup_steps: 0           # Wan LoRA typically doesn't need warmup
  min_lr_ratio: 0.01        # floor at 1% of peak LR

lora:
  rank: 32                  # community standard
  alpha: 16                 # alpha = rank/2 is conservative default
  loraplus_lr_ratio: 4      # validated on Wan, essentially free
  dropout: 0.0              # 0 for <20 clips, 0.05 for larger datasets
```

### Differential MoE Override Defaults

For Wan 2.2's dual-expert architecture, overrides per expert:

```yaml
# High-noise expert (global composition, motion)
expert_high:
  learning_rate: 1e-4       # Minta's experimental finding
  rank: 16
  max_epochs: 30            # converges faster

# Low-noise expert (fine detail, texture)
expert_low:
  learning_rate: 8e-5       # more conservative
  rank: 24
  max_epochs: 50            # needs longer, overfits aggressively
```

### What Dimljus Should Support (Phase 5 Config)

**Must support (validated on Wan):**
- adamw, adamw8bit, adafactor
- constant, cosine, cosine_with_min_lr schedulers
- loraplus_lr_ratio as LoRA network arg

**Should support (community interest, available in musubi-tuner):**
- came_pytorch
- prodigy (prodigyopt.Prodigy)
- rex scheduler
- linear, polynomial schedulers

**Can defer (limited/no Wan validation):**
- AdEMAMix
- Schedule-Free optimizers
- Prodigy-Plus-Schedule-Free

### Config Schema Implications

1. **betas must be variable-length.** Adam uses 2, CAME uses 3. The config should accept a list.
2. **eps must be variable-length.** Adam uses 1 scalar, CAME uses 2. Accept scalar or list.
3. **Optimizer-specific args need a passthrough dict.** Prodigy needs d_coef, safeguard_warmup, etc. CAME needs its triple-beta. Rather than modeling every optimizer's args, provide `optimizer_args: {}` for arbitrary kwargs.
4. **loraplus_lr_ratio lives in LoRA config, not optimizer config.** It creates parameter groups, not a different optimizer.
5. **Per-expert optimizer overrides** are needed for differential MoE. The config should allow overriding optimizer/scheduler/LoRA settings per expert.
6. **Scheduler min_lr needs multiple specification methods:** absolute value (`min_lr: 1e-6`) or ratio (`min_lr_ratio: 0.01`).

---

## References

- CAME: arXiv:2307.02047, GitHub yangluo7/CAME, PyPI came-pytorch
- REX: arXiv:2107.04197, GitHub IvanVassi/REX_LR, MLSys 2022
- LoRA+: arXiv:2402.12354, implemented in musubi-tuner/sd-scripts
- Prodigy: arXiv:2306.06101, GitHub konstmish/prodigy, PyPI prodigyopt
- AdEMAMix: arXiv:2409.03137, GitHub apple/ml-ademamix
- Schedule-Free: arXiv:2405.15682, GitHub facebookresearch/schedule_free, PyPI schedulefree
- Simplified AdEMAMix: arXiv:2502.02431
- Prodigy-Plus-Schedule-Free: GitHub LoganBooker/prodigy-plus-schedule-free, PyPI prodigy-plus-schedule-free
- Musubi-tuner: GitHub kohya-ss/musubi-tuner (advanced_config.md, Discussion #182, #455, #514)
- AI-Toolkit: GitHub ostris/ai-toolkit (config/examples/train_lora_wan21_14b_24gb.yaml)
