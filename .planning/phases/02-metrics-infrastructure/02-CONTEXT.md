# Phase 2: Metrics Infrastructure - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Build automatic logging so every training run from Phase 3 onward captures the data needed to compare experiments. Covers: per-phase loss curves, wall-clock timing, VRAM tracking, per-expert loss divergence, fixed-seed visual samples, and frozen-expert weight verification. Uses the working inference from Phase 1 for sample generation.

</domain>

<decisions>
## Implementation Decisions

### Logging backend
- **Weights & Biases (W&B)** as primary logging backend
- Key metrics (loss, epoch, step, lr) also printed to console during training for quick glance
- Per-expert loss (high-noise, low-noise) logged as **separate W&B panels** — not overlaid on one chart. W&B allows overlaying later if needed, but default view is clean separate panels
- VRAM tracked with **periodic sampling + peak** — log VRAM usage every N steps to W&B (shows memory curve over time), report peak at end of run

### Visual sample settings
- Samples generated **every N epochs** (configurable in training config)
- **Configurable prompt set** — user defines a list of prompts in training config YAML. Dimljus provides a sensible default set, but fully customizable per experiment
- **Default to small resolution** (e.g. 480x832, 17 frames) for speed. Configurable if user wants to override, but matching training resolution would be too expensive as default
- Samples logged to W&B as **both video and keyframe grid image** — scrub the video for detail, glance at the grid for quick convergence check

### Run organization
- **Auto-descriptive run naming** — auto-generate from config: model, dataset, expert mode, key params (e.g. `wan22-holly-unified-r16-lr1e4`). User can override with a custom name
- **Full resolved config saved** with every run — YAML snapshot saved to run directory AND logged to W&B config tab. Any run can be reproduced from its saved config

### Checksum verification
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

</decisions>

<specifics>
## Specific Ideas

- Minta emphasized that sample generation at training resolution would be "crazy" — keep default small for speed, let users opt into higher res
- Fixed-seed samples are the primary way Minta evaluates convergence — this is a visual-first workflow, not a metrics-first one
- W&B already configured as MCP tool in Claude Code — leverage existing integration

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-metrics-infrastructure*
*Context gathered: 2026-02-27*
