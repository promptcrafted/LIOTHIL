# Phase 5 Review Notes

Working document for tracking changes during Minta's file-by-file review.
Started 2026-02-24. Changes applied during review (not deferred).

---

## training_defaults.py → RESTRUCTURED

### Decision: Split into per-model template files
**Problem:** One file tried to hold universal `DEFAULT_*` constants + `MODEL_TEMPLATES` dict for all models. But T2V and I2V are separate training strategies with different values. Shared defaults create a merge-to-understand problem — you can't look at one place and see what a model actually gets. Wan 2.1 is a completely different architecture that shouldn't be here at all.

**Applied fix:**
- Created `wan22_t2v_training_template.py` — everything about Wan 2.2 T2V in one self-contained file
- Removed Wan 2.1 entirely (add back in a future phase)
- Removed Wan 2.2 I2V (gets its own file later: `wan22_i2v_training_template.py`)
- Valid option sets (optimizer names, scheduler names, precision modes) duplicated per template — self-contained > DRY
- Template encodes fork-and-specialize as the PRIMARY MoE strategy
- All numerical values live inside the T2V file with T2V_ prefix
- File named "template" not "defaults" — these are starting points users override, not rigid values

### Chunk-by-chunk changes applied:

**Docstring** — Changed to "experimental" framing ("starting points, not rigid defaults")

**VALID_OPTIMIZERS** — Added `ademamix` (Apple dual-EMA) and `schedule_free_adamw` (Meta). Both untested on Wan but tracked for future use.

**VALID_SCHEDULERS** — Added 5 missing schedulers found in ai-toolkit and musubi-tuner: `constant_with_warmup`, `cosine_with_restarts`, `inverse_sqrt`, `step`, `warmup_stable_decay`. Total: 11 schedulers.

**VALID_FORK_TARGETS** — Expanded from component-level only to two-level targeting:
- Component-level: `ffn`, `self_attn`, `cross_attn` (targets all projections)
- Projection-level: e.g. `cross_attn.to_v`, `ffn.up_proj` (targets specific projection)
- Rules: component listed = all projections train; specific projections listed = only those, rest frozen; component not listed = fully frozen
- Motivated by attn2.to_v.bias showing cosine 0.44 (most divergent individual tensor)

**T2V_LEARNING_RATE** — Changed from `2e-4` to `5e-5`. User feedback: "community reccs are trash for wan 2.2." Conservative default for fork-and-specialize where low-noise expert overfits at high LR.

**T2V_LORA_ALPHA** — Added explicit constant (was implicit via rank). Users need to see both rank and alpha even if they usually keep the relationship.

**T2V_BASE_MODEL_PRECISION** — Changed from `fp8` to `bf16`. User: "I hate quantizing or cheats to speed things up." Quality-first default, user can opt into fp8 if VRAM-constrained.

**T2V_OPTIMIZER** — Kept `adamw8bit` after discussion. This is standard tooling (not a quality shortcut), validated across cloud and local.

**Batch size docstring** — Removed "VRAM-constrained" framing. Neutral language.

**LoRA rank/alpha docstrings** — Added notes that rank = network_dim and alpha = network_alpha in musubi/kohya terminology.

**Per-expert rank overrides** — REMOVED. Rank determines LoRA matrix dimensions (A: d×r, B: r×d). Once created during unified phase, shape is locked. Can't change rank after fork.

**Fork-and-specialize section** — Major additions:
- `T2V_FORK_ENABLED: bool = True` — master switch. False = unified-only training.
- `T2V_UNIFIED_TARGETS` — component targeting during unified phase (None = all)
- `T2V_UNIFIED_BLOCK_TARGETS` — block targeting during unified phase (None = all 40)
- Per-expert `dropout`, `fork_targets`, `block_targets`, `resume_from` fields
- Four training modes documented in comments:
  1. Fork-and-specialize (recommended): fork_enabled=True, unified_epochs > 0
  2. Unified only: fork_enabled=False
  3. Expert-only from scratch: fork_enabled=True, unified_epochs=0
  4. Expert-only from file: fork_enabled=True, unified_epochs=0, resume_from set

**Expert training behavior clarified:** Both experts train interleaved in the same loop via timestep routing. Each step samples a timestep, boundary ratio determines which expert handles it, only that expert's LoRA updates. NOT sequential.

**Full quality audit** — Reviewed entire file for speed-over-quality defaults. Only adamw8bit retained (standard tooling). No other shortcuts found.

**Unified/fork section reorganization** — Moved `T2V_UNIFIED_EPOCHS`, `T2V_UNIFIED_TARGETS`, `T2V_UNIFIED_BLOCK_TARGETS` from the fork section into the Unified Training Loop section. In the Pydantic schema, moved `unified_epochs`, `unified_targets`, `unified_block_targets` from `MoeConfig` into `TrainingLoopConfig`. This cleanly separates unified-phase settings from fork-phase settings, making it straightforward to extract `wan22_t2v_unified_only.py` and `wan22_t2v_expert_only.py` templates later without structural changes.

**Renamed:** `wan22_t2v_training_template.py` → `wan22_training_master.py`. One master file covers both T2V and I2V, and all three training modes. No separate Python files — different modes and variants are just different YAML configurations.

**I2V added to master file.** Only values that differ from T2V are listed (I2V section is compact). Key differences: 36 input channels (reference image), boundary 0.900, unified_epochs 15 (longer — more shared signal), caption_dropout 0.15 (higher — reference image carries conditioning). TEMPLATE_DEFAULTS dict maps template names to override values for the loader.

**YAML examples planned** (same master, different YAML settings):
- `wan22_t2v_fork.yaml` — T2V fork-and-specialize (primary)
- `wan22_t2v_unified.yaml` — T2V unified only (`moe.fork_enabled: false`)
- `wan22_t2v_expert.yaml` — T2V expert only (`training.unified_epochs: 0`)
- `wan22_i2v_fork.yaml` — I2V fork-and-specialize
- (more I2V examples as needed)

**Major restructure: Fixed vs Overridable** — Reorganized Part 2 constants into three zones:
1. **Training Strategy** — `fork_enabled` moved to top as master switch
2. **Fixed Settings** — rank, alpha, loraplus, optimizer type, betas, eps, max_grad_norm, warmup, precision, timestep_sampling. These don't change between unified and expert phases.
3. **Unified Foundation** — LR, weight_decay, scheduler, min_lr_ratio, lora_dropout, batch_size, gradient_accumulation, caption_dropout, unified_epochs, unified_targets, unified_block_targets. These are starting values that each expert can override after fork.

**Removed T2V_MAX_EPOCHS** — Redundant. Duration is unified_epochs (shared phase) + per-expert max_epochs. No need for a user-facing "total epochs" constant.

**Removed T2V_FORK_TARGETS** — Redundant shared default. Fork targets are per-expert and per-unified, not universal. Each expert's fork_targets defaults to None = inherit from unified_targets.

**New per-expert overrides** — Added to MoeExpertOverrides: `batch_size`, `gradient_accumulation_steps`, `caption_dropout_rate`, `weight_decay`, `min_lr_ratio`. All nullable (None = inherit from unified). Motivated by: different experts may need different compute/regularization treatment after fork.

**Scheduler per-expert: confirmed yes** — Different convergence profiles justify per-expert scheduler. High-noise converges faster → more aggressive decay. Low-noise needs gentler treatment. `scheduler_type` already existed; `min_lr_ratio` added.

**Weight decay per-expert: confirmed yes** — Regularization knob. Low-noise overfits more easily → may benefit from higher weight decay. Less commonly tuned than LR but the option should exist.

---

## Pre-Release: Clean Template Version

**TODO before git release:** Create a minimal-annotation version of the master file. Same values, same structure, but stripped down to field names + one-line hints. Ship both:
- `wan22_t2v_training_master.py` — full annotations (the "why" version)
- `wan22_t2v_training_master_clean.py` or similar — minimal comments (the "just configure" version)

Some users want the walls of text. Some don't. Give them both.

---

## training_schema.py → MERGED INTO TEMPLATE

**Status:** Merged into `wan22_t2v_training_template.py`. The separate `training_schema.py` file will be removed.

**Known changes needed** (from template review):
- `ForkAndSpecializeConfig` should not be optional/nullable — fork is primary
- Remove `rank` from `MoeExpertOverrides` (rank can't change after fork)
- Add `fork_targets`, `block_targets` to expert overrides
- Add per-expert `dropout`, `resume_from` fields
- Add `unified_targets`, `unified_block_targets` to fork config
- Add `fork_enabled` boolean
- Schema must support all four training paths
- Revise to reference per-model template files instead of shared defaults

---

## training_loader.py

**Status:** NOT YET REVIEWED

**Known changes needed:**
- Load from per-model template files instead of `MODEL_TEMPLATES` dict
- Template selection maps to a specific template file (e.g. `wan22_t2v` → `wan22_t2v_training_template.py`)

---

## Example YAML files

**Status:** NOT YET REVIEWED

**Known changes needed:**
- Examples should reflect 5e-5 LR, bf16 base precision
- Fork-and-specialize should be shown as primary (not commented-out experimental)
- differential_moe_train.yaml may need updating for fork_targets

---

## tests/test_training_config.py

**Status:** NOT YET REVIEWED

**Known changes needed:**
- Tests must match all schema/loader/template changes above
- Template tests should test per-model files
- Fork-and-specialize tests need expansion for four training paths
