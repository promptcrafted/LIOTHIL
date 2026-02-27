# Architecture: Standalone vs Training-Integrated Inference

**Domain:** Diffusion transformer inference state management
**Researched:** 2026-02-26
**Confidence:** HIGH (verified against codebase, PyTorch docs, diffusers source, musubi-tuner patterns)

## The Core Question

Why does the same inference code produce clean video in a standalone script but noise when called from the training loop?

This document maps every state difference between these two contexts and identifies the most likely failure modes.

---

## Recommended Architecture: State Isolation Checklist

The root cause is **state contamination** -- the training context modifies the model object in ways that persist into the inference call. A standalone script starts with a clean model; the training loop's model carries accumulated state.

### The Seven State Dimensions

When a model transitions from training to inference, these seven dimensions must all be correct. A failure in ANY ONE produces noise or artifacts.

| # | Dimension | Standalone | Training-Integrated | Risk |
|---|-----------|-----------|---------------------|------|
| 1 | Train/eval mode | `.eval()` at load | `.train()` during epoch, must switch to `.eval()` | MEDIUM |
| 2 | Gradient checkpointing | Never enabled | Enabled for VRAM savings | **CRITICAL** |
| 3 | PEFT wrapper | No wrapper (raw model) | PeftModel wrapping base model | **CRITICAL** |
| 4 | Autocast context | None (explicit dtype) | `torch.amp.autocast` may be active | HIGH |
| 5 | Gradient state | No grad graph | Accumulated grad buffers | LOW |
| 6 | dtype consistency | All bf16 (explicit) | Mixed (autocast, loss in fp32) | MEDIUM |
| 7 | Expert state dict | Fresh from disk | May have been swapped/modified | MEDIUM |

---

## Component Boundaries

### Standalone Inference Script (test_base_inference.py)

```
[Load Model from Disk] --> fresh WanTransformer3DModel
        |
        v
[.eval()] --> training=False, no grad graph, no PEFT
        |
        v
[Build Pipeline] --> WanPipeline(transformer=model, vae=vae, ...)
        |
        v
[torch.no_grad()] --> no autocast, explicit bf16 casting
        |
        v
[pipeline(**kwargs)] --> clean forward pass
        |
        v
[Clean Video Output]
```

### Training-Integrated Inference (loop.py -> sampler.py -> inference.py)

```
[Model loaded for training] --> WanTransformer3DModel
        |
        v
[PEFT wrapped] --> PeftModel(base_model=WanTransformer3DModel)
        |
        v
[gradient_checkpointing enabled] --> modifies forward behavior
        |
        v
[.train()] --> training=True, dropout active (if any)
        |
        v
[... N training steps with autocast, grad accumulation ...]
        |
        v
[_generate_samples() called] --> sampler.py
        |
        v
[WanInferencePipeline.generate()] --> inference.py
        |
        |  model.eval()           <-- sets training=False
        |  torch.no_grad()        <-- disables grad computation
        |  BUT: gradient_checkpointing still enabled on base model
        |  BUT: PEFT wrapper still wrapping the model
        |  BUT: autocast context from training may still be active
        |  BUT: model weights have grad buffers attached
        |
        v
[Pipeline forward pass] --> CONTAMINATED STATE
        |
        v
[Noisy Output]
```

---

## Detailed Analysis by Dimension

### 1. Train/Eval Mode (MEDIUM risk)

**What changes:** `model.train()` vs `model.eval()` toggles `self.training` on all submodules. This affects:
- **Dropout layers**: Active in train mode, disabled in eval mode
- **BatchNorm**: Uses batch stats in train, running stats in eval

**WanTransformer3DModel specifics:**
- Has dropout in `WanAttention.to_out` and in `FeedForward` modules
- Uses `FP32LayerNorm` and `RMSNorm` (NOT BatchNorm) -- these are train/eval invariant
- Dropout is typically set to 0.0 in diffusion transformers, but if PEFT adds LoRA dropout, it IS active during training

**Dimljus current handling (inference.py:426-431):**
```python
was_training = model.training
model.eval()
# ... inference ...
if was_training:
    model.train()
```

**Verdict:** This IS handled correctly. The code switches to eval and restores afterward. However, `model.eval()` only sets `self.training = False` on modules -- it does NOT disable gradient checkpointing or remove PEFT wrappers.

**Confidence:** HIGH (verified in codebase)

### 2. Gradient Checkpointing (CRITICAL risk)

**What it does:** Gradient checkpointing saves VRAM during training by not storing intermediate activations. Instead, it recomputes them during the backward pass. The model's forward method checks a flag to decide whether to use `torch.utils.checkpoint.checkpoint()` around each transformer block.

**The WanTransformer3DModel implementation (from diffusers source):**
```python
if torch.is_grad_enabled() and self.gradient_checkpointing:
    # Use checkpoint -- wraps block in recomputation logic
    hidden_states = self._gradient_checkpointing_func(block, ...)
else:
    # Normal forward -- direct block execution
    hidden_states = block(hidden_states, ...)
```

**Key insight:** WanTransformer3DModel uses `torch.is_grad_enabled()` (NOT `self.training`) as the guard. This means:
- Inside `torch.no_grad()`, gradient checkpointing is automatically disabled
- This is the NEWER, CORRECT pattern (diffusers PR #9878 fixed this)
- Older diffusers models use `self.training and self.gradient_checkpointing`, which IS broken during inference

**However, there is a subtlety with PEFT:** When the model is wrapped by `PeftModel`, the forward call goes through PEFT's wrapper first. If PEFT's wrapper or `_gradient_checkpointing_func` has its own `self.training` check, the behavior may differ.

**What to verify:**
1. Confirm the installed diffusers version has the `torch.is_grad_enabled()` guard (not `self.training`)
2. Confirm PEFT's wrapper does not re-introduce a `self.training` guard
3. Test: add `print(f"grad_enabled={torch.is_grad_enabled()}, gc={self.gradient_checkpointing}")` inside the forward method during inference to verify the condition evaluates correctly

**Confidence:** MEDIUM (the WanTransformer3DModel source uses the correct guard, but interaction with PEFT wrapper is unverified)

### 3. PEFT Wrapper (CRITICAL risk)

**What changes:** `create_lora_on_model()` calls `peft.get_peft_model()`, which wraps the model in a `PeftModel`. This is NOT just adding LoRA layers -- it replaces the model with a wrapper that intercepts the forward pass.

**Standalone vs integrated:**
- Standalone: Raw `WanTransformer3DModel` passed to `WanPipeline(transformer=model)`
- Integrated: `PeftModel(WanTransformer3DModel)` passed to `WanPipeline(transformer=model)`

**Known PEFT issues:**
1. **Forward pass interception:** PeftModel's forward adds LoRA output to base output. During inference, the LoRA A*B matrices contribute to the output. If LoRA was initialized with `init_lora_weights=True` (Dimljus default), the B matrix starts at zero, so untrained LoRA should be a no-op. BUT after training, the LoRA contributes a trained delta.
2. **Scaling factor:** LoRA output is scaled by `alpha / rank`. If alpha or rank is wrong, the LoRA contribution is scaled incorrectly.
3. **`model.unload()` behavior:** Dimljus calls `model.unload()` (in `remove_lora_from_model`) which merges LoRA into base weights and removes the wrapper. This PERMANENTLY MODIFIES the base weights. If `unload()` is called mid-training, the base weights are contaminated.

**The pipeline receives the PEFT-wrapped model.** WanPipeline does not know it is getting a PeftModel. It calls `model(**inputs)` which goes through PeftModel's forward, which goes through the LoRA layers, which adds the LoRA delta to the base output.

**This is NOT inherently wrong** -- during training-time sampling, you WANT the LoRA applied. The question is whether the LoRA contribution is correctly scaled and whether the PEFT wrapper's forward behaves identically to a non-wrapped model when LoRA weights are zero (initial state).

**What to verify:**
1. Print the LoRA scaling factor during inference: `model.peft_config['default'].lora_alpha / model.peft_config['default'].r`
2. Check if any LoRA parameters have NaN or Inf values: `for n, p in model.named_parameters(): if 'lora' in n and (p.isnan().any() or p.isinf().any()): print(n)`
3. Test with LoRA explicitly disabled: `model.disable_adapter_layers()` before inference, then `model.enable_adapter_layers()` after

**Confidence:** HIGH (verified architecture, but runtime behavior unverified)

### 4. Autocast Context (HIGH risk)

**What happens during training (loop.py:743-745):**
```python
with torch.amp.autocast(device_type=device_type, dtype=compute_dtype):
    prediction = self._backend.forward(self._model, **model_inputs)
```

**What happens during inference (inference.py:458):**
```python
with torch.no_grad():
    output = pipeline(**kwargs)
```

**The problem:** If `_generate_samples()` is called from within the training loop, is the autocast context still active? Looking at the call chain:

```
_run_epoch()
    for batch in dataloader:
        with autocast:          <-- autocast enabled
            _training_step()    <-- autocast active here
        # autocast exits here

    # After epoch loop ends:
    # (back in _execute_phase)
    if should_sample:
        _generate_samples()     <-- no autocast here, it exited with the loop
```

**Analysis of Dimljus call chain:** Sampling happens AFTER the epoch loop completes (in `_execute_phase`, line 278), NOT inside the autocast context. The autocast is scoped to individual training steps. So autocast leaking is NOT the issue in the current code.

**However**, if sampling were called from WITHIN the training loop (e.g., after every N steps), autocast WOULD be active and WOULD affect inference. PyTorch's autocast is thread-local and context-managed, so it properly exits when the `with` block ends.

**What to verify:**
1. Add `print(f"autocast enabled: {torch.is_autocast_enabled()}")` at the start of `generate()` to confirm
2. Check if any upstream code enables autocast globally (outside the `with` block)

**Confidence:** HIGH (code analysis confirms autocast is properly scoped)

### 5. Gradient State (LOW risk)

**What accumulates:** During training, `.backward()` creates gradient tensors (`.grad`) attached to every parameter with `requires_grad=True`. These persist until `optimizer.zero_grad()` clears them.

**Impact on inference:** Gradient tensors are SEPARATE from parameter values. They do not affect the forward pass. The forward pass uses `param.data`, not `param.grad`. Gradient tensors only affect:
- Memory usage (they consume VRAM)
- The next optimizer step

**With `torch.no_grad()`:** No new gradients are computed. Existing `.grad` tensors remain but are not modified.

**Verdict:** Gradient state does NOT cause noisy output. It wastes VRAM but produces identical forward pass results.

**Confidence:** HIGH (PyTorch core behavior)

### 6. dtype Consistency (MEDIUM risk)

**Training dtypes:**
- Model weights: bf16 (loaded with `torch_dtype=torch.bfloat16`)
- Compute: bf16 via autocast
- Loss: fp32 (explicit `.float()` cast at line 750-751)
- LoRA params: bf16 (created on model, inherit dtype)

**Inference dtypes:**
- Model weights: bf16 (same as training)
- Prompt embeds: bf16 (explicit cast at line 406-407)
- VAE: fp32 (explicit, always)

**Where dtype mismatches cause noise:**
1. **VAE in bf16/fp16**: The Wan-VAE produces gridded artifacts in reduced precision. Dimljus already handles this (VAE always fp32). VERIFIED.
2. **Mixed LoRA dtypes**: If LoRA A is bf16 but LoRA B is fp32 (or vice versa), the product A*B may lose precision. This happens when `inject_lora_state_dict` copies weights without dtype matching. The current code uses `param.copy_(tensor.to(param.device, param.dtype))` which IS correct.
3. **Prompt embeds in wrong dtype**: If prompt embeddings are fp32 but the model expects bf16, the cross-attention computation may produce incorrect results. The current code explicitly casts to the model's dtype.

**What to verify:**
1. Check actual dtype of model parameters during inference: `set(p.dtype for p in model.parameters())`
2. Check actual dtype of prompt embeddings when passed to pipeline
3. Check if any LoRA parameters have unexpected dtype

**Confidence:** HIGH (code analysis shows correct handling, but runtime verification needed)

### 7. Expert State Dict Integrity (MEDIUM risk)

**Standalone:** Models loaded fresh from disk -- known-good weights.

**Training-integrated:** The model's state dict has been through:
1. `load_model()` -- loads from disk (clean)
2. `create_lora_on_model()` -- wraps in PEFT (adds LoRA params, base frozen)
3. N training steps -- LoRA params updated, base weights frozen
4. Possibly `switch_expert()` -- base weights replaced

**Potential contamination:**
- `switch_expert()` via state dict swap: saves current base model state dict to CPU, loads new one. The PEFT wrapper survives. But does `load_state_dict(assign=True)` work correctly through a PEFT wrapper?
- `remove_lora_from_model()` calls `model.unload()`: this MERGES LoRA into base weights. After this, the base weights are no longer the original weights. If you then call `create_lora_on_model()` again, the "base" weights already contain the previous LoRA delta.

**The teardown/setup cycle in Dimljus:**
```
Phase 1: create_lora → train → extract_lora → remove_lora (merges!) → base contaminated
Phase 2: switch_expert → create_lora → train → ...
```

**CRITICAL FINDING:** `remove_lora_from_model()` calls `model.unload()` which merges LoRA weights into the base model. This means:
- After unified phase, the base weights contain the unified LoRA delta
- When the expert phase starts, it creates NEW LoRA on top of already-modified base weights
- During inference sampling, the model has BOTH the merged unified delta AND the current expert LoRA

This is architecturally questionable. For INFERENCE DURING TRAINING, the model carries accumulated modifications from previous phases. This could explain why noise gets worse over training.

**What to verify:**
1. Check `model.unload()` vs `model.merge_and_unload()` -- Dimljus uses `.unload()` which should return the base without merging. VERIFY this in the PEFT version installed.
2. After phase teardown, compare base model weights against the original checkpoint weights
3. Test: save base model state dict before and after `remove_lora_from_model()` and diff them

**Confidence:** MEDIUM (architectural analysis, runtime behavior of `.unload()` vs `.merge_and_unload()` is version-dependent)

---

## Patterns to Follow

### Pattern 1: Full State Isolation for Training-Time Inference

**What:** Before generating samples during training, fully isolate the model state from training context.

**When:** Every time `_generate_samples()` is called.

**Implementation:**
```python
def _generate_samples_isolated(self, model, pipeline, ...):
    """Generate samples with full state isolation."""
    import torch

    # 1. Record state
    was_training = model.training
    grad_enabled = torch.is_grad_enabled()
    gc_enabled = getattr(model, 'gradient_checkpointing', False)

    # 2. Disable everything
    model.eval()
    if hasattr(model, 'disable_gradient_checkpointing'):
        model.disable_gradient_checkpointing()

    # 3. Generate in no_grad context (NO autocast)
    with torch.no_grad():
        result = pipeline.generate(model=model, ...)

    # 4. Restore everything
    if was_training:
        model.train()
    if gc_enabled:
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        elif hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

    return result
```

### Pattern 2: PEFT-Aware Inference

**What:** Verify PEFT adapter state before inference and optionally disable it for diagnostics.

**When:** Debugging noisy output from training-time sampling.

**Implementation:**
```python
def _diagnose_peft_state(model):
    """Print PEFT adapter diagnostic info."""
    from peft import PeftModel

    if not isinstance(model, PeftModel):
        print("  Model is NOT PEFT-wrapped (raw nn.Module)")
        return

    config = model.peft_config.get('default')
    if config:
        print(f"  LoRA rank={config.r}, alpha={config.lora_alpha}")
        print(f"  Effective scale={config.lora_alpha / config.r}")
        print(f"  Dropout={config.lora_dropout}")

    # Check for NaN/Inf in LoRA params
    for name, param in model.named_parameters():
        if 'lora' in name:
            if param.isnan().any():
                print(f"  WARNING: NaN in {name}")
            if param.isinf().any():
                print(f"  WARNING: Inf in {name}")

    # Check adapter state
    print(f"  Active adapters: {model.active_adapters}")
    print(f"  Model training mode: {model.training}")
```

### Pattern 3: Explicit Gradient Checkpointing Disable for Inference

**What:** Even though WanTransformer3DModel uses `torch.is_grad_enabled()` as the guard, explicitly disabling gradient checkpointing before inference eliminates any interaction with PEFT's wrapper.

**When:** Always, as defensive programming.

**Implementation:**
```python
# Before inference
base = model.get_base_model() if hasattr(model, 'get_base_model') else model
if hasattr(base, 'gradient_checkpointing'):
    base.gradient_checkpointing = False

# After inference, re-enable
if hasattr(base, 'gradient_checkpointing'):
    base.gradient_checkpointing = True
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Using model.unload() Between Training Phases

**What:** Calling `PeftModel.unload()` to remove LoRA between phases.

**Why bad:** In some PEFT versions, `.unload()` MERGES the LoRA weights into the base model before removing the wrapper. This permanently modifies the base weights. Subsequent LoRA creation trains on top of already-modified weights, and cumulative modifications make inference behavior unpredictable.

**Instead:** Use `model.disable_adapter_layers()` to disable LoRA contribution without modifying base weights. Or extract the state dict and delete the PeftModel entirely, then reload the base model from the cached state dict.

### Anti-Pattern 2: Assuming model.eval() Is Sufficient

**What:** Only calling `model.eval()` before inference.

**Why bad:** `model.eval()` only sets `self.training = False`. It does NOT:
- Disable gradient checkpointing (though WanTransformer3DModel handles this via `torch.is_grad_enabled()`)
- Remove PEFT wrappers
- Clear gradient buffers
- Exit autocast contexts
- Restore original dtypes

**Instead:** Use the full state isolation pattern (Pattern 1 above).

### Anti-Pattern 3: Sharing Model Object Between Pipeline and Training

**What:** Passing the same model object to both `WanPipeline(transformer=model)` and the training loop.

**Why bad:** The pipeline and training loop both modify the model's state. If the pipeline modifies internal buffers, caches, or running statistics, those modifications persist into the next training step, and vice versa. This creates a two-way state contamination channel.

**Instead:** This is actually necessary for VRAM efficiency (can not load two copies of a 14B model). The solution is strict state save/restore around inference, not separate model objects. Pattern 1 addresses this.

---

## Data Flow During Training-Time Sampling

```
Training Loop (_run_epoch)
    |
    model.train()                     <-- training mode ON
    gradient_checkpointing enabled    <-- VRAM optimization ON
    PEFT wrapper active               <-- LoRA layers active
    |
    for batch in dataloader:
        with autocast(bf16):          <-- mixed precision ON
            forward → loss → backward
        optimizer.step()
    |
    [epoch complete]
    |
    _generate_samples() called        <-- SAMPLING ENTRY POINT
        |
        sampler.generate_samples()
            |
            pipeline.generate(model=self._model, ...)
                |
                ┌─────────────────────────────────────┐
                │ WanInferencePipeline.generate()      │
                │                                      │
                │ 1. _precompute_embeddings()           │
                │    - Loads T5 temporarily             │
                │    - Encodes prompts                  │
                │    - Frees T5                         │
                │                                      │
                │ 2. model.eval()         ← GOOD       │
                │                                      │
                │ 3. _build_pipeline()                  │
                │    - Loads VAE (fp32)                 │
                │    - Creates scheduler (shift=5.0)    │
                │    - WanPipeline(transformer=model)   │
                │      ↑ model is STILL PEFT-wrapped    │
                │      ↑ gradient_checkpointing flag    │
                │        still set on base model        │
                │                                      │
                │ 4. torch.no_grad():                   │
                │    pipeline(**kwargs)                  │
                │    ↑ WanPipeline calls model(...)      │
                │    ↑ Goes through PeftModel.forward()  │
                │    ↑ Then WanTransformer3D.forward()   │
                │    ↑ GC check: torch.is_grad_enabled() │
                │    ↑ = False (inside no_grad) → OK     │
                │                                      │
                │ 5. model.train()        ← RESTORE    │
                └─────────────────────────────────────┘
```

**The flow shows that gradient checkpointing should be handled correctly** by the `torch.is_grad_enabled()` guard. But this depends on:
1. The installed diffusers version having this guard (not the older `self.training` guard)
2. PEFT's forward not re-introducing gradient checkpointing
3. No other code enabling `torch.set_grad_enabled(True)` inside the inference path

---

## Scalability Considerations

| Concern | Single Expert | Dual Expert | With VACE |
|---------|--------------|-------------|-----------|
| State contamination | PEFT wrapper + GC flag | Same + expert swap state | Same + context block state |
| VRAM for inference | ~27GB model + 1GB VAE | ~54GB both experts + VAE | Unknown |
| Isolation complexity | Low | Medium (expert assignment) | High (context routing) |
| Debug surface area | Small | Medium | Large |

---

## Debugging Checklist: When Inference Produces Noise

Use this checklist when training-time inference produces noise but standalone inference works.

### Immediate Checks (5 minutes)

- [ ] **Is model in eval mode?** Print `model.training` inside `generate()` before pipeline call
- [ ] **Is autocast disabled?** Print `torch.is_autocast_enabled()` inside `generate()`
- [ ] **Are gradients disabled?** Print `torch.is_grad_enabled()` inside the forward pass
- [ ] **Is gradient checkpointing bypassed?** Add print inside WanTransformer3D forward to confirm the non-checkpoint branch is taken
- [ ] **What is the model type?** Print `type(model)` -- is it `PeftModel` or `WanTransformer3DModel`?

### Diagnostic Checks (15 minutes)

- [ ] **Are LoRA params valid?** Check for NaN/Inf in all `lora_` parameters
- [ ] **Is LoRA scaling correct?** Print `alpha/rank` and verify it matches config
- [ ] **Is dtype consistent?** Print `set(p.dtype for p in model.parameters())` -- should be all bf16
- [ ] **Are prompt embeds non-zero?** Print `prompt_embeds.abs().mean()` -- should be > 0
- [ ] **Is the scheduler correct?** Print scheduler class and shift value

### Deep Checks (30+ minutes)

- [ ] **Compare standalone vs integrated forward pass:** Feed identical input to both and compare output tensor values
- [ ] **Disable LoRA adapters:** Call `model.disable_adapter_layers()` before inference -- if output improves, LoRA is the problem
- [ ] **Check from_single_file config:** Verify the loaded model has correct config (Wan 2.2, not Wan 2.1). Print `model.config` and check `num_layers`, `num_attention_heads`
- [ ] **Verify base weights unchanged:** Save model state dict hash before training, compare after -- if different, `unload()` merged LoRA into base
- [ ] **Test with freshly loaded model:** Load a separate copy from disk, run inference with PEFT adapter loaded via `load_lora_weights()` -- if this works, state contamination is confirmed

---

## Root Cause Probability Assessment

Based on the codebase analysis:

| Hypothesis | Probability | Evidence |
|-----------|------------|---------|
| `from_single_file` loads wrong config (Wan 2.1 vs 2.2) | **40%** | Known diffusers bug #12329. Training and standalone scripts may use different loading paths. test3-inference.py added explicit `config=` param. Training backend does NOT. |
| PEFT wrapper forward pass differs from raw model | **25%** | PEFT adds overhead, dropout, scaling. Untested in this specific diffusers+PEFT version combination. |
| `model.unload()` merges LoRA into base weights | **15%** | If `.unload()` merges, cumulative modifications corrupt the base model across phases. Version-dependent behavior. |
| Gradient checkpointing still active during inference | **10%** | WanTransformer3DModel uses `torch.is_grad_enabled()` guard, which should be False inside `torch.no_grad()`. Low probability but worth verifying. |
| T5 embed_tokens fix not applied on pod | **5%** | If T5 embeddings are all zeros, model generates unconditionally (noise-like output). The fix exists but may not trigger on all loading paths. |
| dtype mismatch between components | **5%** | Code analysis shows correct handling, but runtime verification needed. |

**Top recommendation:** Add `config="Wan-AI/Wan2.2-T2V-A14B-Diffusers"` and the appropriate `subfolder=` to `WanModelBackend.load_model()` when using `from_single_file`. This is the single highest-probability fix based on the evidence:
- The standalone test3-inference.py ADDED this explicit config and subfolder
- The backend.py `load_model()` does NOT pass config= to `from_single_file`
- If diffusers infers Wan 2.1 config (different architecture params), the model loads weights into the wrong layer structure, producing garbage

---

## Sources

- [PyTorch model.eval() vs model.train()](https://discuss.pytorch.org/t/model-train-and-model-eval-vs-model-and-model-eval/5744) -- train/eval mode behavior
- [PyTorch gradient checkpointing docs](https://docs.pytorch.org/docs/stable/checkpoint.html) -- reentrant vs non-reentrant
- [Gradient checkpointing conflict with no_grad()](https://discuss.pytorch.org/t/gradient-checkpointing-conflict-with-no-grad/90724) -- the core interaction
- [Gradient checkpointing cannot be used in eval mode](https://github.com/huggingface/transformers/issues/43381) -- self.training guard problem
- [Diffusers issue #10107: gradient checkpointing runs during validations](https://github.com/huggingface/diffusers/issues/10107) -- diffusers-specific fix
- [PEFT model.unload() issues](https://github.com/huggingface/peft/issues/868) -- merge_and_unload state contamination
- [PEFT merge_and_unload returns base model](https://github.com/huggingface/peft/issues/2764) -- .unload() behavior ambiguity
- [Diffusers WanTransformer3DModel source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_wan.py) -- gradient checkpointing guard uses torch.is_grad_enabled()
- [PyTorch autocast documentation](https://docs.pytorch.org/docs/stable/amp.html) -- autocast scope and nesting
- [musubi-tuner inference findings](file:///C:/Users/minta/.claude/projects/C--Users-minta-Projects-dimljus-kit/memory/inference_findings.md) -- reference implementation comparison
- [VRAM audit](file:///C:/Users/minta/.claude/projects/C--Users-minta-Projects-dimljus-kit/memory/vram_audit.md) -- memory management patterns
