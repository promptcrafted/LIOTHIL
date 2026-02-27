# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** The inference pipeline must produce recognizable video -- without working sampling, nothing else in the trainer can be validated.
**Current focus:** Phase 2: Metrics Infrastructure — Plan 01 COMPLETE

## Current Position

Phase: 2 of 7 (Metrics Infrastructure)
Plan: 1 of 2 in current phase
Status: Plan 01 complete, Plan 02 pending
Last activity: 2026-02-27 -- Completed 02-01-PLAN.md (core metrics infrastructure)

Progress: [###.......] 21%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: varied
- Total execution time: ~2.2 hours cumulative

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-fix-inference-pipeline | 2 | ~2 hrs | ~1 hr |
| 02-metrics-infrastructure | 1 | 9 min | 9 min |

**Recent Trend:**
- Last 5 plans: 01-01 (4 min), 01-02 (multi-session), 02-01 (9 min)
- Trend: accelerating

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

### Pending Todos

None yet.

### Blockers/Concerns

- RESOLVED: Base model inference produces noisy grid → fixed by config= parameter + T5 embed_tokens fix
- RESOLVED: Scheduler settings conflict → FlowMatchEuler shift=5.0 + boundary=0.6 validated as best
- RESOLVED: from_single_file config detection bug (diffusers#12329) → explicit config= on all calls
- LoRA inference not yet validated (LoRA checkpoint not available for testing)

## Session Continuity

Last session: 2026-02-27
Stopped at: Completed 02-01-PLAN.md (core metrics infrastructure)
Resume file: None
