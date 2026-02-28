---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: in_progress
last_updated: "2026-02-28T03:36:00Z"
progress:
  total_phases: 7
  completed_phases: 2
  total_plans: 10
  completed_plans: 5
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** The inference pipeline must produce recognizable video -- without working sampling, nothing else in the trainer can be validated.
**Current focus:** Phase 3: Expert Isolation Tests -- Plan 1 of 2 COMPLETE

## Current Position

Phase: 3 of 7 (Expert Isolation Tests)
Plan: 1 of 2 in current phase (03-01 COMPLETE, 03-02 remaining)
Status: Plan 03-01 complete, ready for 03-02 (post-training inference validation)
Last activity: 2026-02-28 -- Completed 03-01-PLAN.md (expert isolation training + sampling validation)

Progress: [#####.....] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: varied
- Total execution time: ~5.4 hours cumulative

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-fix-inference-pipeline | 2 | ~2 hrs | ~1 hr |
| 02-metrics-infrastructure | 2 | 15 min | 7.5 min |
| 03-expert-isolation-tests | 1/2 | ~3 hrs | ~3 hrs |

**Recent Trend:**
- Last 5 plans: 01-02 (multi-session), 02-01 (9 min), 02-02 (6 min), 03-01 (multi-session, ~3 hrs incl. GPU training)
- Trend: GPU plans take longer (expected -- real training runs)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Inference fix is hard prerequisite -- nothing validates without it
- config= only needed for from_single_file(), not from_pretrained()
- VAE from_single_file() does not need config= fix -- bug is transformer-only
- T5 must be loaded from HF repo, NOT standalone .pth file
- T5 embed_tokens weight tying fix required for all loading paths
- FlowMatchEuler shift=5.0 + boundary=0.6 is the T2V inference default (best temporal quality, ComfyUI-aligned)
- Training boundary (0.875) differs from inference boundary (0.6) — scheduler-dependent SNR mapping
- Keyframe grid uses frame indices (0,4,8,12,16) for evenly spaced coverage of 17-frame output
- Used family+variant for W&B run name prefix (wan+22t2v=wan22t2v)
- VRAM logging skips console output intentionally -- too frequent for terminal
- Resolved config saved early in run() so config is captured even if training crashes
- File-based checksum is primary frozen-expert verification strategy (no GPU memory needed)
- Frozen expert verification at phase boundaries (before/after entire phase), not per-epoch
- Default T2V prompts cover 4 quality dimensions: static person, motion, scenic, detail
- Wan 2.2 experts are specialists -- MUST load both for inference (single expert produces noise)
- diffusers 0.36.0 WanPipeline regression -- pinned <0.36 until upstream fix
- Flat lr 5e-5 for isolation tests, no per-expert overrides
- Isolation LoRA checkpoints already have diffusers prefix (transformer_2 for low-noise)

### Pending Todos

None yet.

### Blockers/Concerns

- RESOLVED: Base model inference produces noisy grid → fixed by config= parameter + T5 embed_tokens fix
- RESOLVED: Scheduler settings conflict → FlowMatchEuler shift=5.0 + boundary=0.6 validated as best
- RESOLVED: from_single_file config detection bug (diffusers#12329) → explicit config= on all calls
- RESOLVED: LoRA inference not yet validated → validated in 03-01 with dual-expert sampling
- RESOLVED: Single-expert sampling produces noise → fixed with partner model loading (387155c)
- RESOLVED: diffusers 0.36.0 WanPipeline regression → pinned <0.36 (e05f18b)

## Session Continuity

Last session: 2026-02-28
Stopped at: Completed 03-01-PLAN.md (expert isolation training + sampling validation)
Resume file: None
