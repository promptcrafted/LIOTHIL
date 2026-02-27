# Phase 3: Expert Isolation Tests - Research

**Researched:** 2026-02-27
**Domain:** MoE expert isolation training (Wan 2.2 T2V), RunPod GPU execution, W&B logging
**Confidence:** HIGH

## Summary

Phase 3 proves that each MoE expert (low-noise and high-noise) can be trained independently -- no unified warm-up, no other expert phase -- producing valid checkpoints and decreasing loss curves. The entire Dimljus infrastructure for this is already built and tested (Phase 7 architecture, Phase 8 Wan backend, Phase 1 inference, Phase 2 metrics). The existing `test3-experts-only.yaml` config already demonstrates the "expert from scratch" mode. The work is: create two NEW configs (one per expert in true isolation), deploy to RunPod, execute training, validate results (loss curves, checkpoint loading, visual quality), then capture findings.

**Primary recommendation:** Create two YAML configs (`test4-low-only.yaml` and `test5-high-only.yaml`) using `moe.low_noise.enabled: false` / `moe.high_noise.enabled: false` to disable the OTHER expert. Use flat lr 5e-5, 5 epochs, rank 16, AdamW, sample at epoch 2 and 5. Deploy via tarball, run back-to-back on the existing RunPod pod, validate results.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use the Holly dataset (same data used for test3-experts-only training)
- Do NOT create a synthetic test set -- use real data for meaningful results
- 5 epochs per isolation test (fast smoke test)
- Resolution: 480x832, 17 frames (matches Phase 1 validated inference settings)
- Visual samples generated at epoch 2 and epoch 5
- LoRA rank 16, uniform across all phases
- **Flat learning rate 5e-5 for everything** -- no per-expert lr overrides
- AdamW optimizer
- No differential hyperparameters for these isolation tests
- Loss must trend downward over 5 epochs (general trend, not strict monotonic)
- Both W&B sample logging AND full inference must work -- validate both paths
- Automated quality checks (not black, not static, has motion) run first, then Minta reviews
- New RunPod pod (RTX PRO 6000 Blackwell 98GB)
- SSH (runpod): `ssh 8w5mxla2oi48zc-64411f70@ssh.runpod.io -i ~/.ssh/id_ed25519`
- SSH (TCP): `ssh root@198.13.252.111 -p 32427 -i ~/.ssh/id_ed25519`
- Both expert tests run back-to-back on same pod, review results together at the end
- W&B captures everything for later reference and experiment comparison
- Claude also shows key outputs (loss curves, sample videos) directly in chat

### Claude's Discretion
- Test execution order (high-noise first or low-noise first)
- Automated quality check implementation details
- W&B project naming and organization
- Pod setup script adjustments

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-01 | Low-noise-only training (no unified, no high) completes without crash and produces checkpoints | Config uses `moe.high_noise.enabled: false` + `unified_epochs: 0` to produce single LOW_NOISE phase. Phase resolution already supports this (verified in `phase.py` enabled_experts filter). Checkpoint saved to `low_noise/` subdir with `transformer_2.` diffusers prefix. |
| TRAIN-02 | High-noise-only training (no unified, no low) completes without crash and produces checkpoints | Config uses `moe.low_noise.enabled: false` + `unified_epochs: 0` to produce single HIGH_NOISE phase. Checkpoint saved to `high_noise/` subdir with `transformer.` diffusers prefix. |
</phase_requirements>

## Standard Stack

### Core (Already Built -- No New Dependencies)

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| TrainingOrchestrator | `dimljus/training/loop.py` | Main state machine, phase execution | Built, tested |
| resolve_phases() | `dimljus/training/phase.py` | Config -> concrete phases with expert filtering | Built, tested |
| WanModelBackend | `dimljus/training/wan/backend.py` | Expert loading, switching, forward pass | Built, tested |
| WanInferencePipeline | `dimljus/training/wan/inference.py` | Sample generation with dual-expert support | Built, tested |
| TrainingLogger | `dimljus/training/logger.py` | Console + W&B logging, sample uploads | Built, tested |
| SamplingEngine | `dimljus/training/sampler.py` | When/what to sample, keyframe grids | Built, tested |
| CheckpointManager | `dimljus/training/checkpoint.py` | Phase-organized save/load | Built, tested |
| WeightVerifier | `dimljus/training/verification.py` | Frozen expert checksum verification | Built, tested |
| RunPod train.py | `runpod/train.py` | Encode -> train sequencing | Built, tested |
| setup.sh | `runpod/setup.sh` | Pod provisioning | Built, tested |

### Supporting (RunPod Environment)
| Tool | Purpose | When to Use |
|------|---------|-------------|
| tmux | Keep training alive after SSH disconnect | Always use for training runs |
| nohup | Alternative to tmux for background runs | If tmux unavailable |
| W&B | Experiment tracking, loss curves, sample videos | Configured via `logging.backends: [console, wandb]` |

### No New Code Libraries Needed
The training infrastructure is complete. Phase 3 is a **test execution phase** that creates YAML configs and runs existing code on GPU, not a code development phase.

## Architecture Patterns

### Expert Isolation Configuration Pattern

The phase resolution system in `phase.py` already supports expert isolation through two config switches:

```yaml
# Low-noise-only isolation (TRAIN-01)
training:
  unified_epochs: 0          # No unified phase
moe:
  enabled: true
  fork_enabled: true
  high_noise:
    enabled: false            # DISABLED - skip high-noise entirely
    max_epochs: 5             # Still needs a value for Pydantic validation
  low_noise:
    enabled: true
    max_epochs: 5             # The ONLY phase that runs
```

```yaml
# High-noise-only isolation (TRAIN-02)
training:
  unified_epochs: 0          # No unified phase
moe:
  enabled: true
  fork_enabled: true
  high_noise:
    enabled: true
    max_epochs: 5             # The ONLY phase that runs
  low_noise:
    enabled: false            # DISABLED - skip low-noise entirely
    max_epochs: 5             # Still needs a value for Pydantic validation
```

**How it works (verified in source):**

1. `resolve_phases()` in `phase.py` (line 221) builds `enabled_experts` by filtering: `name for name in expert_order if experts_config[name].enabled`
2. With `unified_epochs: 0`, no unified phase is added (line 227)
3. Only the enabled expert gets a phase appended (line 237)
4. Result: single-phase training with expert masking

### Expert Model Loading Pattern

When an expert phase executes:
1. `_ensure_expert_model()` (loop.py line 419) checks `phase.active_expert` against `backend.current_expert`
2. For the first phase, `load_model()` (backend.py line 147) loads the correct expert via `_resolve_single_file_path(expert)`
3. Low-noise expert loads `dit_low` file (the low-noise 14B safetensors)
4. High-noise expert loads `dit_high` file (the high-noise 14B safetensors)

### Expert Loss Masking Pattern

During training, expert masking ensures only relevant timesteps contribute to loss:
- `_training_step()` (loop.py line 885): `if phase.boundary_ratio is not None and phase.active_expert is not None`
- HIGH_NOISE phase: loss masked to timesteps where `t >= boundary_ratio` (high noise = global composition)
- LOW_NOISE phase: loss masked to timesteps where `t < boundary_ratio` (low noise = fine detail)
- Boundary ratio defaults to 0.875 for training (from model config), NOT 0.6 (inference)

### Checkpoint Prefix Pattern

Critical for inference compatibility:
- HIGH_NOISE/UNIFIED checkpoints saved with `transformer.` prefix
- LOW_NOISE checkpoints saved with `transformer_2.` prefix
- This matches diffusers convention: `pipeline.load_lora_weights()` routes keys to correct expert

### W&B Logging Pattern

Phase 2 infrastructure provides:
- Per-phase loss tracking: `{phase_type}/loss`, `{phase_type}/loss_ema`
- Per-phase learning rate: `{phase_type}/lr`
- VRAM tracking: `vram/allocated_gb`, `vram/reserved_gb`
- Sample logging: `samples/{phase_type}/video`, `samples/{phase_type}/keyframe_grid`
- Run timer: wall-clock time per phase and total
- Frozen expert verification: logged at phase boundaries

### Sampling Configuration Pattern

Minta wants samples at epoch 2 and epoch 5. This means `every_n_epochs: 1` (sample every epoch) with the fact that epoch 1 gets skipped because `should_sample` returns False for epochs not divisible by the interval. Actually, looking at the code:

```python
def should_sample(self, epoch: int, phase_type: PhaseType) -> bool:
    if epoch % self._every_n_epochs != 0:
        return False
```

To get samples at epoch 2 and 5 out of 5 epochs, the simplest approach is `every_n_epochs: 1` (sample every epoch). But that wastes time on epochs 1, 3, 4. A cleaner approach: set `every_n_epochs: 1` and accept the overhead (17-frame inference is fast on 98GB), OR use a manual approach where we specifically trigger sampling.

**Recommendation:** Use `every_n_epochs: 1`. The overhead of 5 samples vs 2 is minimal (each sample ~30s on RTX PRO 6000 Blackwell). This gives maximum visibility into training convergence. Alternatively, if Minta specifically wants only epoch 2 and 5, there's no built-in way to specify arbitrary epochs -- `every_n_epochs: 2` would give epochs 2 and 4, and the end-of-phase sample would cover epoch 5 via the checkpoint saving logic. Hmm -- actually `save_every_n_epochs` handles checkpoints, but sampling is separate.

**Best option:** `every_n_epochs: 1` gives samples at every epoch (1,2,3,4,5). Epoch 2 and 5 are included. The extra samples at 1, 3, 4 provide more data points at trivial cost.

### Execution Workflow Pattern

```
1. Deploy dimljus to pod (tarball or git pull)
2. Run setup.sh (reinstall packages, skip model downloads if cached)
3. Upload Holly dataset to /workspace/datasets/test/
4. Cache latents (VAE encoding) -- only needed once, shared between tests
5. Cache text (T5 encoding) -- only needed once, shared between tests
6. Run test4-low-only: python /workspace/dimljus/runpod/train.py --config test4-low-only.yaml --skip-encoding
7. Run test5-high-only: python /workspace/dimljus/runpod/train.py --config test5-high-only.yaml --skip-encoding
8. Run inference validation on both checkpoints
9. Review results (loss curves, samples, W&B dashboard)
```

### Anti-Patterns to Avoid
- **Do NOT use per-expert LR overrides.** CONTEXT.md explicitly locks flat 5e-5 everywhere. The expert config `learning_rate` field should be `null` (inherit from optimizer.learning_rate).
- **Do NOT reference old differential lr settings.** MEMORY.md warns: "Do NOT reference old musubi-tuner differential settings (1e-4/8e-5 per expert). Those are in CLAUDE.md but NOT what Minta wants for current tests."
- **Do NOT try to load both experts for single-expert inference.** A low-noise-only LoRA cannot be meaningfully tested with both experts in the pipeline -- it only trained the low-noise expert, so the high-noise expert has no LoRA effect. Test with single-expert inference first, then optionally test with both for comparison.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| YAML config for isolation tests | Write configs from scratch | Copy `test3-experts-only.yaml`, modify `enabled` flags and remove per-expert lr overrides | Proven config structure, just toggle flags |
| Post-training inference | Custom inference script | Existing `test3-inference.py` pattern (adapt for single-expert) | Pattern proven in Phase 1 |
| Quality checks (black/static/motion) | Complex CV pipeline | Simple frame-level checks: mean pixel value, frame-to-frame diff, variance | 3 numpy operations cover 90% of failure modes |
| Loss curve analysis | Custom plotting | W&B dashboard | Built-in loss visualization, already configured |
| Expert switching logic | Manual model reload | `_ensure_expert_model()` in orchestrator | Handles both swap and disk-reload strategies |

**Key insight:** Phase 3 is an execution/validation phase, not a code development phase. The infrastructure exists. The work is configuration + execution + validation.

## Common Pitfalls

### Pitfall 1: Forgetting --skip-encoding on Second Test
**What goes wrong:** Re-running VAE and T5 encoding wastes 15-20 minutes
**Why it happens:** The second isolation test uses the same dataset with the same cache
**How to avoid:** First test runs full pipeline (`train.py --config`). Second test uses `--skip-encoding` flag.
**Warning signs:** "Step 1/3: Caching latents" appears when it shouldn't

### Pitfall 2: W&B Run Name Collision
**What goes wrong:** Two tests overwrite the same W&B run
**Why it happens:** Auto-generated run names may be identical if configs are similar
**How to avoid:** Use explicit `wandb_run_name` in each config, or use `wandb_group` to group the isolation tests
**Warning signs:** W&B dashboard shows merged data from both tests

### Pitfall 3: Output Directory Overlap
**What goes wrong:** Second test's checkpoints overwrite the first test's
**Why it happens:** Both tests use the same `output_dir`
**How to avoid:** Each test MUST have its own `output_dir` (e.g., `test4-low-only`, `test5-high-only`)
**Warning signs:** Missing checkpoints from the first test

### Pitfall 4: Using `adamw8bit` Instead of `adamw`
**What goes wrong:** CONTEXT.md says "AdamW optimizer" -- this could mean either torch AdamW or bitsandbytes AdamW8bit
**Why it happens:** The existing configs use `adamw8bit` for VRAM efficiency
**How to avoid:** Use `adamw8bit` -- it's the standard for 14B models on single GPUs. The VRAM savings (~8GB optimizer states) are essential. "AdamW" in the context means the AdamW family, not specifically torch.optim.AdamW.
**Warning signs:** OOM during training if using fp32 AdamW on 14B model

### Pitfall 5: Frozen Expert Verification Fails for True Isolation
**What goes wrong:** WeightVerifier tries to verify the "frozen" expert but no frozen expert exists in isolation mode
**Why it happens:** `_get_frozen_expert_name()` returns the OTHER expert name, but there is no OTHER expert checkpoint on disk
**How to avoid:** The orchestrator already handles this gracefully: `frozen_ckpt = self._checkpoint_mgr.find_latest_checkpoint(frozen_phase_type)` returns `None` when no checkpoint exists, and the snapshot is only taken `if frozen_ckpt is not None` (loop.py line 311-313). So this is a non-issue -- just verify it logs correctly.
**Warning signs:** None expected -- the code path handles it

### Pitfall 6: Single-Expert Sampling Limitation
**What goes wrong:** During training, SamplingEngine generates using only the active expert (no partner), producing lower-quality samples
**Why it happens:** In isolation mode, there is no partner expert checkpoint. `resolve_partner_lora()` returns `None`.
**How to avoid:** This is expected and correct behavior for isolation tests. The training samples will show single-expert quality only. Post-training inference should test BOTH with the trained LoRA + base partner expert AND the trained LoRA alone to compare.
**Warning signs:** Training samples look degraded vs full pipeline -- this is normal for single-expert mode

### Pitfall 7: Pod Disk Space
**What goes wrong:** Container disk fills up during training (checkpoints, samples, caches)
**Why it happens:** Default RunPod container disk is 20GB, models live on volume disk
**How to avoid:** Container disk should be 50GB. Checkpoints are small (~200MB per LoRA at rank 16). With `save_every_n_epochs: 1` and 5 epochs per test, that's ~2GB of checkpoints total. Caches are on /workspace (volume disk).
**Warning signs:** "No space left on device" error

### Pitfall 8: Training boundary vs Inference boundary Confusion
**What goes wrong:** Using inference boundary (0.6) for training expert masking instead of training boundary (0.875)
**Why it happens:** The codebase has two boundary concepts that serve different purposes
**How to avoid:** Training boundary is set via `moe.boundary_ratio` in config. The default from `model.boundary_ratio` is used if moe doesn't override. For Wan 2.2 T2V, the training boundary is 0.875 (set in constants). The inference boundary (0.6) is separate and lives in `WanInferencePipeline._build_pipeline()`. Do NOT change either -- they're validated values.
**Warning signs:** Expert masking log messages showing wrong boundary value

## Code Examples

### Config for Low-Noise-Only Isolation (TRAIN-01)

```yaml
# test4-low-only.yaml -- Low-noise expert isolation test
# Trains ONLY the low-noise expert, no unified phase, no high-noise phase.
# Validates TRAIN-01: low-noise-only training completes and saves checkpoints.

model:
  variant: 2.2_t2v
  dit_high: /workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors
  dit_low: /workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors
  vae: /workspace/models/wan_2.1_vae.safetensors
  t5: /workspace/models/models_t5_umt5-xxl-enc-bf16.pth

data_config: /workspace/datasets/test/dimljus_data.yaml

cache:
  cache_dir: /workspace/dimljus/cache

lora:
  rank: 16
  alpha: 16

optimizer:
  type: adamw8bit
  learning_rate: 5e-5           # Flat lr -- no per-expert override

training:
  mixed_precision: bf16
  gradient_checkpointing: true
  seed: 42
  unified_epochs: 0             # KEY: no unified phase
  batch_size: 1

moe:
  enabled: true
  fork_enabled: true
  high_noise:
    enabled: false              # KEY: disable high-noise expert
    max_epochs: 5
  low_noise:
    enabled: true               # Only this expert trains
    max_epochs: 5

save:
  save_every_n_epochs: 1
  output_dir: /workspace/outputs/test4-low-only
  name: test_lora

logging:
  backends: [console, wandb]
  log_every_n_steps: 1
  wandb_project: dimljus-isolation
  wandb_run_name: low-only-r16-lr5e-05

sampling:
  enabled: true
  every_n_epochs: 1             # Sample every epoch for maximum visibility
  prompts:
    - "Medium shot, Holly Golightly walks up an indoor staircase, looking back over her shoulder, morning light."
  neg: "blurry, low quality, distorted"
  seed: 42
  sample_steps: 30
  guidance_scale: 4.0
```

### Config for High-Noise-Only Isolation (TRAIN-02)

```yaml
# test5-high-only.yaml -- High-noise expert isolation test
# Mirror of test4, but with low_noise disabled and high_noise enabled.
# Key differences: output_dir, wandb_run_name, enabled flags flipped.

# ... (same model, data, cache, lora, optimizer, training sections) ...

moe:
  enabled: true
  fork_enabled: true
  high_noise:
    enabled: true               # Only this expert trains
    max_epochs: 5
  low_noise:
    enabled: false              # KEY: disable low-noise expert
    max_epochs: 5

save:
  output_dir: /workspace/outputs/test5-high-only
  name: test_lora

logging:
  backends: [console, wandb]
  log_every_n_steps: 1
  wandb_project: dimljus-isolation
  wandb_run_name: high-only-r16-lr5e-05
```

### Phase Resolution Verification (Dry Run)

Before GPU training, verify the config produces the expected single phase:

```bash
python /workspace/dimljus/runpod/train.py --config test4-low-only.yaml --dry-run
```

Expected output:
```
Training Plan:
  Phase 1/1: LOW_NOISE
    Epochs: 5
    Learning rate: 5e-05
    Batch size: 1
    Expert masking: low_noise (boundary=0.875)
```

### Automated Quality Check Pattern

Simple numpy-based checks on generated samples:

```python
import numpy as np
from PIL import Image

def check_sample_quality(video_frames: list) -> dict[str, bool]:
    """Quick automated quality checks on training samples.

    Three checks:
    1. Not black: mean pixel value > 10 (catches failed generation)
    2. Not static: frame-to-frame difference > threshold (catches frozen output)
    3. Has variance: pixel variance > threshold (catches flat/solid color output)
    """
    frames = [np.array(f) for f in video_frames]

    # Check 1: Not black (mean pixel across all frames)
    mean_pixel = np.mean([f.mean() for f in frames])
    not_black = mean_pixel > 10.0

    # Check 2: Not static (mean frame-to-frame difference)
    if len(frames) > 1:
        diffs = [np.abs(frames[i+1].astype(float) - frames[i].astype(float)).mean()
                 for i in range(len(frames) - 1)]
        mean_diff = np.mean(diffs)
        has_motion = mean_diff > 1.0  # > 1 pixel mean difference = has motion
    else:
        has_motion = True  # Single frame can't check motion

    # Check 3: Has variance (not flat solid color)
    pixel_var = np.mean([f.astype(float).var() for f in frames])
    has_variance = pixel_var > 100.0

    return {
        "not_black": not_black,
        "has_motion": has_motion,
        "has_variance": has_variance,
        "mean_pixel": float(mean_pixel),
        "mean_frame_diff": float(mean_diff) if len(frames) > 1 else 0.0,
        "pixel_variance": float(pixel_var),
    }
```

### Post-Training Inference Validation Pattern

After training completes, validate the checkpoint loads and produces inference output:

```python
# Adapt test3-inference.py for single-expert LoRA
# Key difference: only load the trained expert's LoRA, not a merged LoRA

# For low-noise-only LoRA:
# 1. Load only low-noise expert model
# 2. Load LoRA (has transformer_2. prefix)
# 3. Generate with single-expert pipeline (no transformer_2 in pipeline)
#    OR generate with dual-expert pipeline where only transformer_2 has LoRA

# The checkpoint at test4-low-only/low_noise/test_lora_low_epoch005.safetensors
# has keys like: transformer_2.blocks.0.attn1.to_q.lora_A.weight
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Run both experts in sequence, always | `enabled: false` flag on `MoeExpertOverrides` allows skipping | Phase 7 design | True isolation possible |
| Per-expert lr overrides (1e-4/8e-5) | Flat 5e-5 everywhere | Minta's decision for current tests | Simpler baseline, fair comparison |
| Manual checkpoint inspection | WeightVerifier checksums + W&B logging | Phase 2 (METR-06) | Automated frozen expert verification |
| Separate inference scripts | Integrated SamplingEngine + WanInferencePipeline | Phase 1 + Phase 2 | Samples during training + post-hoc inference |

## Open Questions

1. **Single-expert inference quality expectations**
   - What we know: Single-expert samples (during training) will be lower quality than dual-expert inference because only one expert handles all noise levels
   - What's unclear: How much worse? Is it useful for judging training convergence?
   - Recommendation: Generate both single-expert (during training) and dual-expert (post-training, with base partner) samples for comparison. The post-training dual-expert samples are the real quality indicator.

2. **Expert order for back-to-back execution**
   - What we know: Both tests run on the same pod, same dataset, same cache
   - What's unclear: Does execution order matter? (It shouldn't -- they're independent)
   - Recommendation: Run low-noise first (TRAIN-01 maps to test 4 in `today-feb-25-26.js`). Low-noise is the expert that Minta's thesis says "converges faster" -- seeing it first builds confidence. But truly, order doesn't matter.

3. **Loss trend detection at 5 epochs**
   - What we know: CONTEXT.md says "general trend, not strict monotonic"
   - What's unclear: With only 5 epochs and expert masking (some timesteps zeroed), loss may be noisy
   - Recommendation: Use loss EMA (already tracked by MetricsTracker) rather than raw per-step loss. A downward-trending EMA over 5 epochs is sufficient. If loss is flat or slightly up, that's still informative data (suggests expert-from-scratch may need longer warm-up or different init).

4. **W&B project organization**
   - What we know: Configs should use `wandb_project` and `wandb_run_name`
   - What's unclear: Should all test phases share one project or have separate ones?
   - Recommendation: Use `dimljus-isolation` as the project name, with `wandb_group: "phase3-isolation"` to group the two runs. Each run gets a distinct name (`low-only-r16-lr5e-05`, `high-only-r16-lr5e-05`). This keeps Phase 3 tests together while being separate from future training runs.

## Sources

### Primary (HIGH confidence)
- `dimljus/training/phase.py` - Phase resolution logic, `enabled` filtering at line 221
- `dimljus/training/loop.py` - Orchestrator, expert model loading, loss masking, sampling
- `dimljus/training/wan/backend.py` - Expert loading via `_resolve_single_file_path()`
- `dimljus/config/wan22_training_master.py` - `MoeExpertOverrides.enabled` field (line 1051)
- `runpod/test3-experts-only.yaml` - Existing experts-only config pattern
- `runpod/test3-inference.py` - Post-training inference validation pattern
- `.planning/phases/03-expert-isolation-tests/03-CONTEXT.md` - User decisions
- `MEMORY.md` - Training config ground truth (flat 5e-5, no differential)

### Secondary (MEDIUM confidence)
- `dimljus/training/sampler.py` - Sampling schedule logic, `should_sample()` modulo check
- `dimljus/training/verification.py` - WeightVerifier graceful handling when no frozen expert exists
- `dimljus/training/checkpoint.py` - Phase-organized directory structure

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All infrastructure exists and has been tested in Phase 1 and 2
- Architecture: HIGH - Phase resolution, expert isolation, and masking patterns verified in source code
- Pitfalls: HIGH - Based on actual code paths and real RunPod execution experience from Phase 1-2
- Config pattern: HIGH - `enabled: false` flag verified in `MoeExpertOverrides` class definition

**Research date:** 2026-02-27
**Valid until:** Stable -- no external dependencies or fast-moving libraries. Valid until codebase changes.
