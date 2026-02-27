# Phase 3: Expert Isolation Tests - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove each MoE expert (low-noise and high-noise) can be trained independently in isolation — no unified warm-up, no other expert phase. Each isolation test must run to completion, save a loadable checkpoint, show decreasing loss, and produce visual output that passes automated quality checks and Minta's review.

</domain>

<decisions>
## Implementation Decisions

### Test dataset
- Use the Holly dataset (same data used for test3-experts-only training)
- Do NOT create a synthetic test set — use real data for meaningful results

### Training scale
- 5 epochs per isolation test (fast smoke test)
- Resolution: 480x832, 17 frames (matches Phase 1 validated inference settings)
- Visual samples generated at epoch 2 and epoch 5

### Training config
- LoRA rank 16, uniform across all phases
- **Flat learning rate 5e-5 for everything** — no per-expert lr overrides. The base lr applies equally to unified, high-noise, and low-noise phases
- AdamW optimizer
- No differential hyperparameters for these isolation tests

### Success criteria
- Loss must trend downward over 5 epochs (general trend, not strict monotonic)
- Both W&B sample logging AND full inference must work — validate both paths
- Automated quality checks (not black, not static, has motion) run first, then Minta reviews outputs that pass

### Execution environment
- New RunPod pod (RTX PRO 6000 Blackwell 98GB)
- SSH (runpod): `ssh 8w5mxla2oi48zc-64411f70@ssh.runpod.io -i ~/.ssh/id_ed25519`
- SSH (TCP): `ssh root@198.13.252.111 -p 32427 -i ~/.ssh/id_ed25519`
- Previous pod is being shut down — all previous work is committed locally
- Both expert tests run back-to-back on same pod, review results together at the end
- Same pod will be reused for Phases 3-6 baseline tests; Phase 7 (production training) gets a fresh pod

### Results delivery
- W&B captures everything for later reference and experiment comparison
- Claude also shows key outputs (loss curves, sample videos) directly in chat for immediate review

### Claude's Discretion
- Test execution order (high-noise first or low-noise first)
- Automated quality check implementation details
- W&B project naming and organization
- Pod setup script adjustments

</decisions>

<specifics>
## Specific Ideas

- The `today-feb-25-26.js` file outlines the full test sequence: tests 4 and 5 in that list correspond to this phase's isolation tests (low-noise only and high-noise only, both without unified)
- Tests 1-3 and 6-9 from that file map to other phases (full pipeline, unified-only, checkpoint resume, still frames)
- The production training (test 10) maps to Phase 7

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-expert-isolation-tests*
*Context gathered: 2026-02-27*
