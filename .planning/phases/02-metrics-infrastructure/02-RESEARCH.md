# Phase 2: Metrics Infrastructure - Research

**Researched:** 2026-02-27
**Domain:** Training observability — W&B logging, VRAM tracking, wall-clock timing, visual sample logging, weight integrity verification
**Confidence:** HIGH

## Summary

Phase 2 builds the automatic logging layer that makes every training run from Phase 3 onward self-documenting. The existing codebase already has solid foundations: `TrainingLogger` supports console/TensorBoard/W&B backends, `MetricsTracker` accumulates per-phase loss with EMA smoothing, and `SamplingEngine` handles fixed-seed visual sample generation with keyframe grids. The work is primarily about **enriching** what these systems log — not rebuilding them.

The six requirements break into three categories: (1) metric logging enhancements (METR-01 per-phase loss, METR-04 per-expert divergence), (2) system instrumentation (METR-02 wall-clock, METR-03 VRAM), and (3) validation/verification (METR-05 visual samples to W&B, METR-06 frozen-expert checksums). All use W&B as the primary backend per Minta's decision, with console as the always-on fallback.

**Primary recommendation:** Enhance the existing `TrainingLogger` and `TrainingOrchestrator` rather than introducing new modules. Add a `VRAMTracker` utility class, a `WeightVerifier` utility class, upgrade W&B init with `define_metric()` for proper axis/summary behavior, and wire sample video+grid logging into W&B via `wandb.Video` and `wandb.Image`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Weights & Biases (W&B)** as primary logging backend
- Key metrics (loss, epoch, step, lr) also printed to console during training for quick glance
- Per-expert loss (high-noise, low-noise) logged as **separate W&B panels** — not overlaid on one chart. W&B allows overlaying later if needed, but default view is clean separate panels
- VRAM tracked with **periodic sampling + peak** — log VRAM usage every N steps to W&B (shows memory curve over time), report peak at end of run
- Samples generated **every N epochs** (configurable in training config)
- **Configurable prompt set** — user defines a list of prompts in training config YAML. Dimljus provides a sensible default set, but fully customizable per experiment
- **Default to small resolution** (e.g. 480x832, 17 frames) for speed. Configurable if user wants to override, but matching training resolution would be too expensive as default
- Samples logged to W&B as **both video and keyframe grid image** — scrub the video for detail, glance at the grid for quick convergence check
- **Auto-descriptive run naming** — auto-generate from config: model, dataset, expert mode, key params (e.g. `wan22-holly-unified-r16-lr1e4`). User can override with a custom name
- **Full resolved config saved** with every run — YAML snapshot saved to run directory AND logged to W&B config tab. Any run can be reproduced from its saved config
- Frozen-expert weight checksums reported as **pass/fail in console + W&B** — print "Frozen expert check: PASS (weights unchanged)" at end of run. Log checksums to W&B. **Fail loudly** if weights changed unexpectedly

### Claude's Discretion
- W&B project name and workspace organization
- W&B run grouping strategy (groups vs tags vs both)
- Local artifact directory structure (checkpoints, samples per run)
- End-of-run console summary format and content
- Warning/anomaly detection strategy (when to surface warnings vs log silently)
- VRAM sampling interval (every N steps — pick something reasonable)
- Default sample prompt set contents
- Default sample generation frequency (every N epochs — pick a sensible default)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| METR-01 | Loss curves logged per training phase (unified, low-noise, high-noise) | Existing `MetricsTracker` already separates per-phase loss. Logger already dispatches to W&B. Need: `define_metric()` setup for proper per-phase panels, and ensure separate W&B sections for each phase type. |
| METR-02 | Training wall-clock time logged per test run | Standard `time.perf_counter()` in Python stdlib. Wrap orchestrator `run()` and each `_execute_phase()`. Log to W&B summary + console end-of-run report. |
| METR-03 | VRAM usage tracked during training | `torch.cuda.memory_allocated()` + `torch.cuda.max_memory_allocated()` for periodic sampling and peak. Log as W&B time series + console peak report. |
| METR-04 | Per-expert loss divergence tracked | Same metric infrastructure as METR-01 but with both expert phases active. Use `define_metric()` with separate panel prefixes (`high_noise/loss_ema`, `low_noise/loss_ema`). Already partially supported by `MetricsTracker.get_all_metrics()`. |
| METR-05 | Fixed-seed visual samples generated at regular intervals during training | `SamplingEngine` already handles this. Need: log generated videos to W&B via `wandb.Video(path)` and grids via `wandb.Image(path)`. Wire into logger's sample event. |
| METR-06 | Expert weight checksums verified (frozen expert weights don't change during other expert's training) | New `WeightVerifier` utility: compute SHA-256 of state dict tensors before/after phase. Report pass/fail to console + W&B summary. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| wandb | >= 0.17 | Experiment tracking, metric logging, video/image logging | Minta's locked decision. Already in `TrainingLogger._init_wandb()`. Industry standard for ML experiment tracking. |
| torch (cuda) | >= 2.0 | VRAM monitoring via `torch.cuda.memory_allocated()` / `max_memory_allocated()` | Already a project dependency. Only reliable way to track PyTorch-managed GPU memory. |
| hashlib | stdlib | SHA-256 checksums for weight verification | Python stdlib, zero dependencies. Fast enough for state dict checksums. |
| time | stdlib | `time.perf_counter()` for wall-clock timing | Python stdlib. `perf_counter` gives highest-resolution monotonic clock. |
| pyyaml | >= 6.0 | Save resolved config as YAML snapshot | Already a core dependency. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pillow | >= 9.0 | Load keyframe grid PNGs for `wandb.Image()` | Already used by `SamplingEngine._save_keyframe_grid()`. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| wandb | TensorBoard | Already supported as backend in TrainingLogger, but Minta locked W&B as primary. TensorBoard stays as optional secondary. |
| hashlib SHA-256 | torch checksum utils | hashlib is stdlib, no PyTorch dependency. We need to hash the raw bytes anyway. |
| time.perf_counter | time.monotonic | perf_counter has higher resolution on Windows (sub-microsecond). |

**Installation:**
```bash
pip install wandb>=0.17
```
Note: `wandb` should be added to `pyproject.toml` as an optional dependency in the `training` extra group, since it is only needed during training runs.

## Architecture Patterns

### Recommended Changes to Existing Structure
```
dimljus/training/
├── logger.py          # MODIFY — enhance W&B init, add video/image/config logging
├── metrics.py         # MINOR MODIFY — possibly add wall-clock tracking fields
├── vram.py            # NEW — VRAMTracker utility (periodic sampling + peak)
├── verification.py    # NEW — WeightVerifier (checksum frozen experts)
├── loop.py            # MODIFY — wire in VRAM tracking, timing, checksums, config save
├── sampler.py         # MINOR MODIFY — return paths for W&B logging
└── ... (existing files unchanged)
```

### Pattern 1: W&B Run Initialization with define_metric()
**What:** Set up W&B with proper metric organization at run start so per-phase loss curves get their own panels automatically.
**When to use:** At `TrainingLogger.__init__()` when W&B backend is active.
**Example:**
```python
# Source: Context7 /wandb/wandb — define_metric docs
import wandb

run = wandb.init(
    project=project,
    name=run_name,       # Auto-generated or user override
    config=resolved_config_dict,  # Full resolved config
    group=group_name,    # Optional grouping
    tags=tags,           # Optional tags
    notes=notes,         # Optional notes
)

# Define per-phase metric axes — each gets its own W&B panel
run.define_metric("global_step")

# Unified phase metrics
run.define_metric("unified/*", step_metric="global_step")
run.define_metric("unified/loss_ema", summary="min")

# High-noise expert metrics
run.define_metric("high_noise/*", step_metric="global_step")
run.define_metric("high_noise/loss_ema", summary="min")

# Low-noise expert metrics
run.define_metric("low_noise/*", step_metric="global_step")
run.define_metric("low_noise/loss_ema", summary="min")

# System metrics
run.define_metric("system/*", step_metric="global_step")
run.define_metric("system/vram_allocated_gb", summary="max")
```

### Pattern 2: VRAM Periodic Sampling
**What:** Sample GPU memory at regular intervals during training and report peak at run end.
**When to use:** Every N training steps inside the inner loop.
**Example:**
```python
import torch

class VRAMTracker:
    """Tracks GPU memory usage during training."""

    def __init__(self, device: int = 0, sample_every_n_steps: int = 50):
        self._device = device
        self._sample_interval = sample_every_n_steps
        self._samples: list[float] = []

    def sample(self, global_step: int) -> dict[str, float] | None:
        """Sample current VRAM if at interval. Returns metrics dict or None."""
        if global_step % self._sample_interval != 0:
            return None
        if not torch.cuda.is_available():
            return None

        allocated = torch.cuda.memory_allocated(self._device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self._device) / (1024**3)
        self._samples.append(allocated)

        return {
            "system/vram_allocated_gb": allocated,
            "system/vram_reserved_gb": reserved,
        }

    def peak(self) -> float:
        """Return peak allocated VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(self._device) / (1024**3)
```

### Pattern 3: Weight Checksum Verification
**What:** Compute SHA-256 of a model's state dict tensors before and after a phase to verify frozen weights were not modified.
**When to use:** Before and after each expert phase where the other expert should be frozen.
**Example:**
```python
import hashlib
from typing import Any

def compute_state_dict_checksum(state_dict: dict[str, Any]) -> str:
    """SHA-256 checksum of all tensor bytes in a state dict."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if hasattr(tensor, "cpu"):
            # PyTorch tensor — get raw bytes
            hasher.update(tensor.cpu().numpy().tobytes())
        elif hasattr(tensor, "tobytes"):
            # NumPy array
            hasher.update(tensor.tobytes())
    return hasher.hexdigest()
```

### Pattern 4: Auto-Descriptive Run Naming
**What:** Generate a descriptive run name from the training config.
**When to use:** When `wandb_run_name` is None (auto mode).
**Example:**
```python
def generate_run_name(config) -> str:
    """Auto-generate a descriptive run name from config."""
    parts = []

    # Model variant
    variant = config.model.variant or "wan"
    parts.append(variant.replace(".", "").replace("_", ""))

    # Dataset name (from data_config filename or save.name)
    parts.append(config.save.name.replace("dimljus_lora", "default"))

    # Training mode
    if config.moe.enabled and config.moe.fork_enabled:
        if config.training.unified_epochs > 0:
            parts.append("fork")
        else:
            parts.append("expert")
    else:
        parts.append("unified")

    # Key params
    parts.append(f"r{config.lora.rank}")
    lr_str = f"{config.optimizer.learning_rate:.0e}".replace("+", "")
    parts.append(f"lr{lr_str}")

    return "-".join(parts)
```

### Pattern 5: Logging Visual Samples to W&B
**What:** After generating sample videos and keyframe grids, log them to W&B.
**When to use:** After `SamplingEngine.generate_samples()` returns paths.
**Example:**
```python
import wandb

def log_samples_to_wandb(
    video_paths: list[Path],
    phase_type: str,
    epoch: int,
) -> None:
    """Log sample videos and their keyframe grids to W&B."""
    for i, video_path in enumerate(video_paths):
        grid_path = video_path.with_suffix(".grid.png")
        log_dict = {}

        # Log video
        if video_path.is_file():
            log_dict[f"samples/{phase_type}/prompt_{i}/video"] = wandb.Video(
                str(video_path), caption=f"epoch {epoch}"
            )

        # Log keyframe grid
        if grid_path.is_file():
            log_dict[f"samples/{phase_type}/prompt_{i}/grid"] = wandb.Image(
                str(grid_path), caption=f"epoch {epoch}"
            )

        if log_dict:
            wandb.log(log_dict)
```

### Anti-Patterns to Avoid
- **Logging too frequently:** Do NOT log VRAM at every step. The W&B overhead (serialization + network) compounds. Every 50 steps is sufficient.
- **Blocking on W&B:** All W&B logging should be fire-and-forget. The training loop should never wait on W&B network calls. Use `wandb.log()` which is already async.
- **Checksumming live PEFT model:** For frozen-expert verification, checksum the *non-LoRA base model weights*, not the PEFT wrapper. The PEFT wrapper adds LoRA parameters that are supposed to change.
- **Logging full-resolution videos:** Sample videos should be 480x832 (17 frames, ~1s) at most. Do not generate at training resolution for logging purposes.
- **Calling wandb.init() more than once:** The existing `TrainingLogger._init_wandb()` calls `wandb.init(reinit=True)`. This is correct. But `define_metric()` must be called right after init, before any `wandb.log()`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Experiment tracking dashboard | Custom web UI | W&B dashboard | Minta already uses W&B. Rich visualization for free. |
| GPU memory monitoring | Parse `nvidia-smi` output | `torch.cuda.memory_allocated()` / `max_memory_allocated()` | PyTorch's own tracking is authoritative for PyTorch allocations. `nvidia-smi` includes non-PyTorch processes. |
| Video/image media logging | Custom file upload to cloud | `wandb.Video()` / `wandb.Image()` | W&B handles encoding, streaming, and in-browser playback. |
| Config serialization | Custom JSON dumper | Pydantic `model_dump()` + PyYAML | Pydantic already handles nested config serialization. |
| Metric charting | matplotlib during training | W&B auto-generated charts via `define_metric()` | W&B generates interactive charts from logged scalars. No code needed. |

**Key insight:** W&B already provides everything needed for metric visualization, media logging, and config tracking. The implementation work is about *wiring* the existing training loop to W&B correctly, not building visualization tools.

## Common Pitfalls

### Pitfall 1: W&B step counter conflicts with phase transitions
**What goes wrong:** When training transitions from unified to high-noise phase, the global step counter continues. W&B logs everything on the global step axis, so phase-specific metrics appear as one continuous line crossing phase boundaries.
**Why it happens:** W&B uses a single monotonic step counter per run by default.
**How to avoid:** Use `define_metric("unified/*", step_metric="global_step")` etc. The per-phase prefix (`unified/`, `high_noise/`, `low_noise/`) naturally separates the curves into different panels. Each phase's metrics only get logged during that phase, so there are no overlapping step ranges.
**Warning signs:** All three phase losses appearing on the same chart.

### Pitfall 2: VRAM measurement timing
**What goes wrong:** VRAM measured at the wrong point in the training step (before forward pass, during gradient accumulation) gives misleading numbers.
**Why it happens:** VRAM fluctuates dramatically within a single training step: peak during forward pass (activations held), drops after backward (activations freed), spikes again during optimizer step (momentum buffers).
**How to avoid:** Sample VRAM *after* each optimizer step (post-backward, post-optimizer-update). This captures the steady-state working memory. Separately track `max_memory_allocated()` for absolute peak.
**Warning signs:** VRAM numbers that seem too low (measured before forward) or wildly variable.

### Pitfall 3: Checksum performance on large state dicts
**What goes wrong:** Checksumming a 14B parameter state dict takes minutes if done naively (moving every tensor to CPU, converting to numpy).
**Why it happens:** The full Wan 2.2 model has ~27B parameters. Even if we only checksum the frozen expert, that is ~14B params.
**How to avoid:** Do NOT checksum the full base model. Instead, checksum only the LoRA-relevant layer outputs or a representative subset. Better approach: snapshot the frozen expert's checkpoint path and verify the file hash (the `.safetensors` file on disk), OR compare a small number of sentinel parameters (first tensor, last tensor, random middle tensor) rather than the full state dict. Since the frozen expert is never loaded into the PEFT model (it stays on disk or in a separate model slot), verifying the checkpoint file is sufficient.
**Warning signs:** Training pausing for 30+ seconds at phase boundaries for checksumming.

### Pitfall 4: W&B offline mode on RunPod
**What goes wrong:** RunPod pods may not have stable internet. W&B logging fails silently or crashes.
**Why it happens:** W&B tries to sync to cloud in real-time.
**How to avoid:** The existing `TrainingLogger` already wraps W&B calls in try/except and suppresses repeated errors. Additionally, `wandb.init(mode="offline")` can be used to log locally and sync later. The config should support this.
**Warning signs:** W&B warning messages in console output, missing data in W&B dashboard.

### Pitfall 5: Visual sample VRAM contention
**What goes wrong:** Generating visual samples during training requires inference VRAM (VAE decoding, T5 encoding) which competes with training VRAM.
**Why it happens:** The `WanInferencePipeline` loads the VAE temporarily and frees it after generation. But during generation, both training model activations and VAE are in VRAM.
**How to avoid:** This is already handled by `WanInferencePipeline` (loads VAE temporarily, frees after). The sampling config defaults to off (`enabled: false`). When enabled, the small resolution (480x832, 17 frames) minimizes extra VRAM. The training loop wraps sampling in try/except so failures do not crash training.
**Warning signs:** OOM errors during sampling epochs.

## Code Examples

### W&B Config Logging — Save Full Resolved Config
```python
# Source: W&B docs — wandb.init config parameter
import wandb
import yaml
from pathlib import Path

def save_and_log_config(config, output_dir: Path) -> None:
    """Save resolved config to disk and log to W&B."""
    # Pydantic v2 model_dump() gives a nested dict
    config_dict = config.model_dump()

    # Save YAML to disk (reproducibility)
    config_path = output_dir / "resolved_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # Log to W&B config tab
    if wandb.run is not None:
        wandb.config.update(config_dict)
```

### Wall-Clock Timing — Phase and Total
```python
import time

class RunTimer:
    """Simple wall-clock timer for phases and total run."""

    def __init__(self):
        self._run_start: float = 0.0
        self._phase_start: float = 0.0
        self._phase_times: dict[str, float] = {}

    def start_run(self) -> None:
        self._run_start = time.perf_counter()

    def start_phase(self, phase_name: str) -> None:
        self._phase_start = time.perf_counter()

    def end_phase(self, phase_name: str) -> float:
        elapsed = time.perf_counter() - self._phase_start
        self._phase_times[phase_name] = elapsed
        return elapsed

    def total_elapsed(self) -> float:
        return time.perf_counter() - self._run_start

    @property
    def phase_times(self) -> dict[str, float]:
        return dict(self._phase_times)
```

### End-of-Run Console Summary
```python
def print_run_summary(
    timer: RunTimer,
    peak_vram_gb: float,
    phase_losses: dict[str, float],
    frozen_check: dict[str, bool],
) -> None:
    """Print a formatted end-of-run summary to console."""
    print("\n" + "=" * 60)
    print("  DIMLJUS TRAINING COMPLETE")
    print("=" * 60)

    # Timing
    total = timer.total_elapsed()
    print(f"\n  Total time: {total/60:.1f} min ({total:.0f}s)")
    for name, t in timer.phase_times.items():
        print(f"    {name}: {t/60:.1f} min")

    # Loss
    print(f"\n  Final loss (EMA):")
    for name, loss in phase_losses.items():
        print(f"    {name}: {loss:.6f}")

    # VRAM
    print(f"\n  Peak VRAM: {peak_vram_gb:.2f} GB")

    # Frozen expert check
    if frozen_check:
        print(f"\n  Frozen expert verification:")
        for name, passed in frozen_check.items():
            status = "PASS" if passed else "FAIL (weights changed!)"
            print(f"    {name}: {status}")

    print("=" * 60 + "\n")
```

### Logging Samples to W&B with Video + Grid
```python
# Source: W&B docs — wandb.Video, wandb.Image
import wandb
from pathlib import Path

def log_samples_to_wandb(
    sample_paths: list[Path],
    phase_type: str,
    epoch: int,
    global_step: int,
) -> None:
    """Log sample videos and keyframe grids to W&B."""
    if wandb.run is None:
        return

    log_dict: dict = {}
    for i, video_path in enumerate(sample_paths):
        if video_path.is_file():
            log_dict[f"samples/{phase_type}/prompt_{i}"] = wandb.Video(
                str(video_path), caption=f"epoch {epoch}", fps=16
            )
        grid_path = video_path.with_suffix(".grid.png")
        if grid_path.is_file():
            log_dict[f"grids/{phase_type}/prompt_{i}"] = wandb.Image(
                str(grid_path), caption=f"epoch {epoch}"
            )

    if log_dict:
        wandb.log(log_dict, step=global_step)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TensorBoard only | W&B + TensorBoard | W&B standard since ~2021 | W&B provides cloud sync, media logging, comparison tools. TensorBoard remains as local fallback. |
| `wandb.log(step=N)` global step only | `define_metric(step_metric=)` per-metric axes | W&B 0.14+ (2023) | Each metric can have its own x-axis (epoch, step, etc.). Critical for per-phase panels. |
| `torch.cuda.memory_stats()` | `torch.cuda.memory_allocated()` + `max_memory_allocated()` | Stable since PyTorch 1.4 | Simple, reliable. `memory_stats()` gives more detail but more complex. |
| `nvidia-smi` parsing | PyTorch CUDA memory API | Always available in PyTorch | PyTorch API is authoritative for PyTorch-managed memory. |

**Deprecated/outdated:**
- `wandb.config` as module-level attribute: replaced by `run.config` instance attribute pattern. Both still work but `run.config` is preferred.
- `wandb.log({"step": N})` as x-axis: replaced by `define_metric(step_metric=)` for proper chart behavior.

## Open Questions

1. **VRAM sampling interval default**
   - What we know: Too frequent (every step) adds overhead. Too infrequent misses VRAM spikes.
   - What's unclear: Optimal balance for 14B model training.
   - Recommendation: Default to every 50 steps. This gives ~1 sample per minute at typical training speed. Configurable in LoggingConfig if needed.

2. **Frozen expert checksum strategy**
   - What we know: The frozen expert is either on disk (checkpoint file) or in a separate model slot. PEFT wraps only the active model.
   - What's unclear: Whether the frozen expert's weights are ever in GPU memory during the other expert's phase (they should not be, but need to verify).
   - Recommendation: Checksum the checkpoint file on disk (fast, no GPU memory needed). If the frozen expert stays on disk during the other's training (which it should per the `_ensure_expert_model` logic), the file checksum is sufficient.

3. **W&B project naming convention**
   - What we know: Minta uses W&B. Multiple experiments will be compared.
   - Recommendation: Default project name `"dimljus"`. Users can override via `logging.wandb_project`. Use W&B groups to cluster related runs (e.g., same dataset, different hyperparameters).

4. **Default sample generation frequency**
   - What we know: Too frequent slows training (full inference per sample set). Too infrequent misses convergence moments.
   - Recommendation: Default `every_n_epochs: 5` (already the default in `SamplingConfig`). For a 30-epoch expert phase, this gives 6 sample checkpoints — enough to track convergence without excessive overhead.

5. **Default sample prompt set**
   - What we know: Needs to cover a range of scenarios: static, motion, close-up, wide shot.
   - Recommendation: 3-4 prompts that test different capabilities:
     - A simple static scene (tests basic generation quality)
     - A motion prompt (tests temporal coherence)
     - A close-up / detail prompt (tests fine detail)
     - Keep prompts generic (not character-specific) since these are defaults

## Sources

### Primary (HIGH confidence)
- Context7 `/wandb/wandb` — `wandb.init()`, `wandb.log()`, `define_metric()`, `wandb.Video`, `wandb.Image` API patterns
- [PyTorch CUDA Memory Documentation](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — `memory_allocated()`, `max_memory_allocated()` API
- [W&B Video Documentation](https://docs.wandb.ai/ref/python/data-types/video/) — Video logging from file path
- [W&B init Documentation](https://docs.wandb.ai/ref/python/init/) — name, group, tags, notes, config parameters
- [W&B Config Documentation](https://docs.wandb.ai/models/track/config) — Nested config dict handling

### Secondary (MEDIUM confidence)
- Existing codebase analysis: `dimljus/training/logger.py`, `metrics.py`, `loop.py`, `sampler.py` — verified current implementation state
- Python stdlib docs — `hashlib.sha256`, `time.perf_counter()` — stable APIs, well-documented

### Tertiary (LOW confidence)
- None — all findings verified against official sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — W&B is a locked decision, PyTorch CUDA memory API is stable and well-documented
- Architecture: HIGH — building on existing well-structured codebase (logger, metrics, sampler already exist)
- Pitfalls: HIGH — common patterns verified against official docs and existing code analysis

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (stable domain, no fast-moving dependencies)
