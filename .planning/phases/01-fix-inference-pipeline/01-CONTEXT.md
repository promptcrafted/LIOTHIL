# Phase 1: Fix Inference Pipeline - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Make base model (Wan 2.2 T2V) and LoRA inference produce recognizable video — not noise, not grids, not static frames. The same inference code must work both standalone and from the training pipeline's sample generation. This is the hard prerequisite for all downstream validation.

</domain>

<decisions>
## Implementation Decisions

### Debugging strategy
- Start by diffing current code against last known-good commit (d2c236d) to find what diverged
- Prioritize investigating the model loading path first — known diffusers bug (#12329) where `from_single_file()` loads wrong config (Wan 2.1 vs 2.2) is the most likely root cause
- If diff + model loading fix doesn't resolve: try `from_pretrained` with full HF repo "Wan-AI/Wan2.2-T2V-A14B-Diffusers" as quick test
- If `from_pretrained` also fails: build minimal stock diffusers reference pipeline as ground truth
- Escalation order: diff → model loading → from_pretrained → fresh reference

### Inference parameters (LOCKED — do not change)
- shift=5.0, boundary_ratio=0.5 for inference — validated by Minta, treat as hard constraints
- FlowMatchEulerDiscreteScheduler — not UniPC
- These are non-negotiable. If something doesn't work, the bug is in the code plumbing, not the parameters
- Training boundary_ratio=0.875 is intentionally different from inference boundary_ratio=0.5

### Validation criteria
- Quality bar: recognizable content + temporal coherence + prompt relevance (not just "not noise")
- Test prompts: 2-3 diverse prompts chosen by Claude (person, scene, motion variety)
- Output spec: 17 frames at 480x832 (current test script defaults, avoids OOM)
- Output format: save .mp4 video + PNG keyframe grid (frames 1, 5, 9, 13, 17) for quick visual review
- No intermediate diagnostics unless actively debugging a specific step

### Reference comparison
- Primary reference: ai-toolkit (source available locally in `ai-toolkit-source/`)
- Match ai-toolkit's code plumbing: model loading sequence, scheduler initialization, denoising loop structure
- DO NOT override Minta's hyperparameters with ai-toolkit's values — if there's a discrepancy, use Minta's
- Only flag if our code is MISSING a parameter that ai-toolkit sets (a gap, not a difference)
- Document what ai-toolkit does differently at each step — useful reference for future debugging
- Key distinction: Minta owns hyperparameters and training methodology. Claude owns code implementation and engineering decisions.

### LoRA test approach
- Use existing test3 checkpoints (expert-only training on Holly) — already trained, saves time
- Success bar: any visible difference from base model output on same prompt = LoRA is working (3 epochs isn't enough for strong likeness)
- Flag if the difference is meaningfully Holly-like (trending toward her appearance), but don't require it
- Generate side-by-side comparisons: identical prompt + seed, once without LoRA, once with LoRA
- Fix standalone inference first, then verify training-pipeline integration (isolate variables)

### Claude's Discretion
- Specific test prompts to use (2-3 diverse prompts)
- Keyframe grid layout and formatting
- Order of investigation within the model loading path
- How to structure the ai-toolkit comparison documentation
- Engineering decisions about code structure and implementation approach

</decisions>

<specifics>
## Specific Ideas

- "I'm not an engineer, and so when it comes to choices about code or infrastructure, I will not be reliable" — defer all engineering decisions to Claude, but never override validated hyperparameters
- Minta's domain: hyperparameters, scheduler type, training methodology, quality evaluation
- Claude's domain: code plumbing, implementation patterns, debugging approach, architecture
- If ai-toolkit uses a parameter value that differs from Minta's, it's informational only — always use Minta's values

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-fix-inference-pipeline*
*Context gathered: 2026-02-26*
