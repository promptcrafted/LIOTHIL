# Feature Landscape: Video LoRA Training Validation Matrix

**Domain:** Video LoRA trainer validation / integration testing
**Researched:** 2026-02-26
**Overall confidence:** HIGH (grounded in existing codebase analysis + competitor trainer patterns + community-reported failure modes)

## Table Stakes

Features users (and developers) expect from any trainer validation suite. Missing any of these means you cannot trust that training actually works.

### Training Correctness Tests

| Test | Why Expected | Complexity | Notes |
|------|-------------|------------|-------|
| Loss decreases over epochs | Minimum proof that gradient updates are moving in the right direction | Low | Already validated in tests 1-3. Log loss_ema per phase, expect monotonic downward trend within a phase. A flat loss = broken gradient flow. |
| Checkpoint saves at expected intervals | Checkpoints are the only durable output of training | Low | Already validated. Verify file size (rank 16, alpha 16 for Wan 2.2 14B = ~293MB per expert). Empty or zero-byte = broken serialization. |
| Checkpoint key format is correct | LoRA must load in standard diffusers/ComfyUI pipelines or it is useless | Med | Validate: diffusers-prefix keys (transformer.blocks.N...), correct LoRA A/B pair count, correct rank dimensions. Already tested in test3 via key counting. |
| Merged LoRA contains both experts | Differential MoE produces TWO expert LoRAs that must merge correctly | Med | Already validated: 800 transformer. + 800 transformer_2. keys = 1600 total in merged file. |
| Gradient norm stays bounded | Exploding gradients produce garbage; vanishing gradients produce no learning | Low | Already tracked via MetricsTracker.grad_norm. Expect values in 0.01-1.0 range after clipping (max_grad_norm=1.0 default). Log and alert if consistently at clip ceiling. |
| Learning rate follows schedule | LR too high = instability, too low = no learning | Low | Already tracked. Verify schedule shape matches config (constant, cosine, etc.). |
| Seed reproducibility | Same config + same seed + same dataset = same loss curve | Med | Run identical config twice, compare loss at step N. Exact match not required (GPU non-determinism) but curves should be nearly identical. |

### Inference Correctness Tests

| Test | Why Expected | Complexity | Notes |
|------|-------------|------------|-------|
| Base model (no LoRA) produces recognizable video | If base inference is broken, nothing else can be validated -- THIS IS THE CURRENT BLOCKER | High | Currently failing (noise/grid output). Must produce coherent frames with recognizable objects/scenes. Visual inspection is the ground truth here. |
| LoRA inference differs from base | Proof the LoRA weights actually affect generation | Med | Same seed + same prompt: base output vs LoRA output must be visually different. Pixel-level L2 distance > threshold, or CLIP embedding distance > threshold. |
| LoRA inference resembles training data | The whole point of training -- the model learned the subject/style | High | Generate with training-similar prompts, visually verify subject/style presence. No automated metric replaces human judgment here, but CLIP-score against reference images provides a sanity floor. |
| Fixed seed produces identical output across runs | Reproducible inference is required for A/B comparisons | Low | Same pipeline + same seed + same prompt = identical frames. Byte-level comparison of output tensors before VAE decode. |
| Output frames have valid pixel range | Noise, NaN, or clamped-to-zero outputs indicate pipeline bugs | Low | Check: min/max in [0, 255] for uint8, no NaN, mean not near 0 or 255, reasonable standard deviation (~30-80 for natural images). Already partially done in test_base_inference.py via inspect_output(). |
| Temporal coherence across frames | Video frames must flow -- frozen or flickering frames mean broken generation | Med | Compare adjacent frames: SSIM > 0.7 (too low = flickering), SSIM < 0.999 (too high = frozen). Optical flow magnitude should be non-zero. |

### Checkpoint Resume Tests

| Test | Why Expected | Complexity | Notes |
|------|-------------|------------|-------|
| Resume from unified checkpoint continues training | Long runs crash. If resume is broken, all progress is lost. | Med | Tests 6-8 on Minta's checklist. Resume at epoch N, verify: (a) loss does not spike catastrophically, (b) loss continues downward trend after brief warm-up, (c) correct epoch numbering. |
| Resume from expert checkpoint continues expert-only training | Expert phases can be very long (50 epochs for low-noise). Must be resumable. | Med | Resume mid-expert-phase. Verify expert isolation is maintained (only the correct expert's LoRA is being updated). |
| Optimizer state loads correctly on resume | Without optimizer state, momentum/adaptive LR is lost and training effectively restarts | High | Compare loss curve at resume point: with optimizer state should show minimal spike, without optimizer state will show a large spike (this is the exact bug musubi-tuner users report in issue #667). |
| Epoch counter continues correctly on resume | Affects when checkpoints save, when sampling triggers, when phases end | Low | Resume at epoch 2, verify next checkpoint saves as epoch 3, not epoch 1. |
| LR scheduler continues correctly on resume | Wrong LR on resume = training diverges or stalls | Med | Log LR at resume point; must match where the schedule would be at that step count. Community reports this is a common bug (musubi-tuner issue #424). |

### Mixed Dataset Tests

| Test | Why Expected | Complexity | Notes |
|------|-------------|------------|-------|
| Still images train as single-frame "videos" | Mixed image+video datasets are standard practice -- images for identity, videos for motion | Med | Test 9 on Minta's checklist. 5 frames from holly_clips/references treated as 1-frame videos. Verify: (a) cache encodes correctly (1 temporal token), (b) loss computes without error, (c) no shape mismatch during training. |
| Video clips of different frame counts in same batch | Real datasets have variable-length clips after scene detection | Med | Verify bucketing handles mixed lengths without padding artifacts. The encoding pipeline should already handle this via resolution/frame buckets. |
| Captions of varying length train correctly | Some captions are 10 words, some are 150 words. T5 encoding must handle both. | Low | Verify padding/truncation works, no silent truncation of long captions. |

## Differentiators

Tests specific to Dimljus's differential MoE architecture and Minta's production methodology. No other trainer validates these because no other trainer has these features.

### Differential MoE Validation

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Per-expert loss curves diverge appropriately | Proves the MoE architecture is doing its job -- experts specialize | High | High-noise expert should converge faster on lower loss (composition is easier). Low-noise expert should have higher initial loss but eventually reach lower final loss (detail work is harder but more precise). If both curves are identical, expert splitting is not working. |
| Expert isolation during training | Each expert's LoRA must update independently -- cross-contamination defeats the purpose | High | During high-noise phase, low-noise LoRA weights must be FROZEN (zero gradient). Verify by checksumming low-noise weights before and after a high-noise epoch -- they must be identical. |
| Expert boundary correctly masks timesteps | Wrong boundary = experts train on the wrong noise levels | High | Training boundary=0.875: verify high-noise expert only sees timesteps > 875/1000, low-noise only sees timesteps < 875/1000. Log timestep distribution per expert per epoch. |
| Unified-to-fork produces correct initialization | The fork point determines expert starting weights | Med | After unified phase, both expert LoRAs must contain identical copies of the unified LoRA. Verify with weight comparison at fork point. |
| Different hyperparams per expert | The core Dimljus insight -- high-noise needs different LR/rank/epochs than low-noise | Med | Verify optimizer is rebuilt per phase with correct LR. Verify epoch count matches config per expert. Already partially tested. |
| Merged LoRA produces better output than unified-only | The whole thesis: differential > unified | High | Test 10 (real training): compare unified-only output vs merged differential output on same prompts. This is the ultimate validation -- requires visual judgment from Minta. |

### Production Quality Validation

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Low-noise overfitting detection | Low-noise experts overfit rapidly (washed-out, loss of dynamic expressions) -- Minta's documented finding | High | Monitor: (a) low-noise loss drops too fast then plateaus, (b) generated samples show loss of motion/expression diversity, (c) training samples from low-noise phase show decreasing visual quality despite decreasing loss. Loss can decrease while visual quality degrades. |
| Sampling with partner LoRA resolution | During expert training, samples must use BOTH experts for accurate preview | Med | Already implemented in SamplingEngine.resolve_partner_lora(). Verify: during high-noise training, samples use current high-noise state + last low-noise checkpoint. Visual confirmation that samples show the full pipeline, not just one expert. |
| VRAM tracking per phase | Production runs need to be predictable -- OOM at epoch 47 of 50 is catastrophic | Med | Log peak VRAM per phase. Verify no VRAM leak across epochs (peak should be stable, not increasing). Already partially tracked via torch.cuda.memory_allocated() in inference scripts. |
| Training time per epoch logging | Required for estimating real training runs (unified 10 + low 15 + high 50 = how many hours?) | Low | Log wall-clock time per epoch per phase. Essential for planning and billing (RunPod is pay-per-hour). |

## Anti-Features

Tests that are overkill for this scope. Explicitly NOT building these.

| Anti-Feature | Why Avoid | What to Do Instead |
|-------------|-----------|-------------------|
| Automated FVD/FID quality scoring | Recent research (arxiv 2410.05203) shows FVD is unreliable for video -- insensitive to temporal distortions, requires impractical sample sizes, non-Gaussian feature space. Would cost significant engineering for unreliable results. | Visual inspection by Minta (the domain expert) remains ground truth. Use simple proxy metrics (pixel range, SSIM between frames) as sanity checks only. |
| CLIP-score automated quality gates | CLIP measures text-image alignment, not video quality or temporal coherence. A frozen frame perfectly matching the prompt would score high. | Use CLIP-score as a floor check (score > 0.15 for relevance) but never as a quality gate. Minta's judgment is the gate. |
| A/B testing framework with statistical significance | This is a training toolkit, not a research platform. The sample sizes (10-clip dataset) are too small for statistical power. | Minta visually compares outputs side-by-side. Log enough metadata (seed, prompt, checkpoint, config) to make comparisons reproducible. |
| Automated regression testing on generated video | Would require a reference video corpus and perceptual similarity metrics that don't exist reliably for video. | Compare against musubi-tuner output on same dataset (Minta's known-good baseline). Manual visual comparison. |
| Multi-GPU distributed training validation | Not in scope. Dimljus targets single-GPU (RunPod pods). Distributed adds complexity for zero current benefit. | Single-GPU validation only. |
| I2V (image-to-video) validation | T2V validation first (Minta's explicit sequencing decision). I2V adds reference image control signal complexity. | Defer to post-T2V milestone. |
| VACE context block validation | Future Phase 10 scope. No VACE training support exists yet. | Defer to Phase 10 milestone. |
| Hyperparameter sweep automation | Premature optimization. Minta has tested hyperparams manually for thousands of hours. The tool should support her settings, not search for new ones. | Fixed configs per test. Minta adjusts manually based on results. |

## Feature Dependencies

```
Base model inference (BLOCKER) --> LoRA inference --> Side-by-side comparison
                                                 --> LoRA quality validation
                                                 --> Differential MoE comparison

Checkpoint save/load --> Checkpoint resume tests (6-8)
                     --> Expert isolation verification

Per-expert loss logging --> Expert divergence validation
                        --> Overfitting detection
                        --> Differential MoE thesis validation

Mixed dataset caching --> Still image training test (9)

All training tests --> Real training run (test 10: unified 10 + low 15 + high 50)
```

## MVP Recommendation

### Immediate priority (unblock everything else):

1. **Fix base model inference** -- This is the single blocker. Nothing else can be validated until the pipeline produces recognizable video. The fix likely involves `from_single_file` config detection (diffusers#12329) or T5 embedding correctness. All other tests depend on this.

### Once inference works:

2. **Run test 4 (low-noise only) and test 5 (high-noise only)** -- These isolate each expert independently. If one works and the other doesn't, the bug is in expert routing, not the pipeline.

3. **Run checkpoint resume tests (6-8)** -- These validate the training loop's durability. Community evidence (musubi-tuner issues #667, #424, #776) shows resume is where most trainers break. Test for: loss spike magnitude, optimizer state restoration, LR continuity.

4. **Run still image test (9)** -- Quick validation of mixed dataset support. Needed before real training since production datasets always mix images and video.

5. **Run real training (10)** -- The ultimate validation. unified 10 + low 15 + high 50, with the differential hyperparams from CLAUDE.md. This proves the thesis or reveals where it breaks.

### Defer:

- **Automated quality metrics**: Not worth engineering time. Minta's visual judgment is faster and more reliable.
- **Cross-trainer comparison automation**: Manual side-by-side with musubi-tuner output is sufficient.
- **Temporal coherence metrics**: Nice-to-have sanity check but not blocking any decisions.

## Metrics to Log and Compare Across Runs

### Per-Step Metrics (already implemented in MetricsTracker)

| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| loss_raw | Instantaneous loss per step | > 10x epoch average = gradient spike |
| loss_ema | Smoothed training progress | Increasing over 100 steps = training diverging |
| grad_norm | Gradient health | Consistently at clip ceiling (1.0) = LR too high |
| learning_rate | Schedule correctness | Deviates from expected schedule = bug |

### Per-Epoch Metrics (need to add)

| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| epoch_loss_mean | Average loss for the entire epoch | Higher than previous epoch = overfitting or instability |
| epoch_loss_std | Loss variance within epoch | Increasing std = training becoming unstable |
| epoch_wall_time_seconds | Wall clock time per epoch | > 2x previous epoch = possible VRAM swap or other bottleneck |
| peak_vram_bytes | Maximum VRAM usage during epoch | Increasing across epochs = memory leak |
| samples_processed | Number of training samples in epoch | Mismatch with dataset size = dataloader bug |

### Per-Phase Metrics (need to add)

| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| phase_start_loss | Loss at phase beginning | For expert phases: should be near unified end loss (if forked) or random (if from scratch) |
| phase_end_loss | Loss at phase completion | Must be lower than start for healthy training |
| phase_total_steps | Total optimizer steps in phase | Must match config: epochs * (samples / batch_size) |
| phase_total_time_seconds | Wall clock time for entire phase | Use for RunPod cost estimation |
| expert_weight_checksum | Hash of frozen expert's weights | Changes during partner's training phase = isolation bug |

### Per-Run Summary Metrics (need to add)

| Metric | What It Tells You | Format |
|--------|-------------------|--------|
| total_training_time | End-to-end duration | HH:MM:SS |
| total_steps_all_phases | Sum of all optimizer steps | Integer |
| final_merged_lora_size | Output file size in MB | Float |
| final_merged_lora_key_count | Number of LoRA weight tensors | Integer (expect 1600 for dual-expert rank 16) |
| config_hash | SHA256 of training config | For reproducibility tracking |

### Comparison Across Runs

For the test matrix, the key comparisons are:

| Comparison | What It Validates | How to Compare |
|-----------|-------------------|----------------|
| Test 1 (full pipeline) vs Test 3 (experts-only) | Does unified warmup help? | Compare phase_end_loss for each expert, compare sample quality |
| Test 4 (low only) vs Test 5 (high only) | Do experts learn independently? | Each should converge; loss curves should differ in shape |
| Test 1/2/3 loss curves vs musubi-tuner on same data | Is Dimljus training correctly? | Loss magnitudes should be in the same ballpark (within 2x) |
| Resumed run vs continuous run | Is resume working? | Loss at resume point should be within 10% of continuous run at same step count, after 100-step warmup |

## How Existing Trainers Validate Output Quality

### musubi-tuner (kohya-ss)

**Validation approach:** Periodic sampling during training via `--sample_every_n_epochs` / `--sample_every_n_steps`. Generates video/image from a prompt file with configurable seed, steps, guidance, and resolution. Saves .mp4/.png to output directory for manual review. No automated quality metrics.

**Resume validation:** Saves full training state (model weights, optimizer state, scheduler state, dataloader sampler state, random state). Known issues: loss spikes on resume due to average loss reset and non-reproduced dataset ordering (issue #667). Workaround: save at epoch boundaries.

**MoE support:** None. musubi-tuner treats Wan 2.2 as a single model; no per-expert training, no expert splitting, no differential hyperparameters.

**Key lesson for Dimljus:** musubi-tuner's community has extensively documented that loss alone is unreliable for quality -- sampling with matching inference parameters (same shift, same guidance) is essential. "Loss values alone don't indicate quality" (Discussion #182).

**Confidence:** HIGH (direct from GitHub docs and issues)

### ai-toolkit (ostris)

**Validation approach:** Periodic sampling with fixed seeds (walk_seed off = same noise every checkpoint for direct visual comparison of training progress). Configurable per-model sampling parameters. Aggressive offloading pipeline for sampling on limited VRAM.

**MoE support:** Wan 2.2 via DualWanTransformer3DModel. `switch_every` parameter alternates between experts during training (default 10 steps). Produces TWO separate LoRA files. Training boundary: 0.875 for T2V, 0.9 for I2V.

**Key difference from Dimljus:** ai-toolkit interleaves expert training steps (switch every N steps), while Dimljus sequences entire phases (unified -> fork -> low N epochs -> high M epochs). ai-toolkit does NOT support different hyperparameters per expert.

**Key lesson for Dimljus:** ai-toolkit's fixed-seed approach for training samples is the gold standard for visual progress tracking. Dimljus already implements walk_seed in SamplingEngine but should also support walk_seed=False for direct frame-by-frame comparison across checkpoints.

**Confidence:** HIGH (direct from ostris tweets, GitHub, DeepWiki documentation)

## Sources

- [musubi-tuner GitHub repository](https://github.com/kohya-ss/musubi-tuner)
- [musubi-tuner sampling during training docs](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/sampling_during_training.md)
- [musubi-tuner Wan 2.2 training discussion #455](https://github.com/kohya-ss/musubi-tuner/discussions/455)
- [musubi-tuner Wan training rules of the trade #182](https://github.com/kohya-ss/musubi-tuner/discussions/182)
- [musubi-tuner resume loss spike issue #667](https://github.com/kohya-ss/musubi-tuner/issues/667)
- [musubi-tuner resume issue #424](https://github.com/kohya-ss/musubi-tuner/issues/424)
- [ai-toolkit GitHub repository](https://github.com/ostris/ai-toolkit)
- [ai-toolkit Wan 2.2 advanced features (DeepWiki)](https://deepwiki.com/ostris/ai-toolkit/13-wan-2.2-advanced-features)
- [ostris Wan 2.2 T2I training announcement](https://x.com/ostrisai/status/1956819166830199215)
- [ostris Wan 2.2 I2V training announcement](https://x.com/ostrisai/status/1957500762843673045)
- [Beyond FVD: Enhanced Evaluation Metrics for Video Generation Quality (arxiv 2410.05203)](https://arxiv.org/html/2410.05203v1)
- [Wan 2.2 official repository](https://github.com/Wan-Video/Wan2.2)
- [RunComfy Wan 2.2 T2V LoRA training guide](https://www.runcomfy.com/trainer/ai-toolkit/wan-2-2-t2v-14b-lora-training)
- [LTX-Video LoRA training study (single image training)](https://huggingface.co/blog/neph1/ltx-lora)
- [WaveSpeedAI Wan 2.2 LoRA training settings](https://wavespeed.ai/blog/posts/blog-wan-2-2-lora-training-settings/)
