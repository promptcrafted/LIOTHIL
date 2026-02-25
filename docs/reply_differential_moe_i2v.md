The short answer is: the reference image is already doing the high-noise expert's job.

Wan 2.2's MoE is two fully separate 14B transformers routed by timestep — the high-noise expert handles early denoising (global composition, motion, camera), and the low-noise expert handles late denoising (identity, texture, fine detail). They're not shared layers with different heads — completely independent parameter sets.

When I ran a weight divergence analysis comparing the two experts across models, the I2V numbers told a clear story:

| Comparison | Cosine Similarity |
|---|---|
| T2V experts vs each other | 0.917 |
| I2V experts vs each other | 0.936 |
| T2V vs I2V high-noise expert | 0.984 |
| T2V vs I2V low-noise expert | 0.949 |

The I2V experts are about 40% less divergent than the T2V experts. And the cross-model comparison is the key detail — the high-noise expert barely changed between T2V and I2V (0.984, nearly identical), while the low-noise expert shifted significantly more (0.949).

That makes sense once you think about what the reference image does. In I2V, the reference image enters through two pathways simultaneously — CLIP ViT-H/14 embeddings via cross-attention, and VAE-encoded latents via channel concatenation (16 image channels + 4 mask channels alongside the 16 noisy video channels). That's strong compositional guidance injected directly into the model. The high-noise expert's primary job — establishing global composition and spatial layout — is already being handled by the image conditioning. It doesn't need to adapt much because the reference image is carrying that burden.

The low-noise expert, on the other hand, has to learn to match fine detail to the reference — preserving identity, texture consistency, subtle features. That's information-dense work. It diverged more because it has more to learn.

So for LoRA training: lower rank on the high-noise expert reflects the fact that it genuinely needs less correction. The reference image provides the compositional scaffold. Higher rank on the low-noise expert gives it the capacity to capture fine detail. And the low-noise expert gets a gentler learning rate (8e-5 vs 1e-4) because it's extremely sensitive to overfit — I found that aggressive parameters on the low-noise expert produce washed-out artifacts, loss of texture fidelity, and failure on dynamic expressions.

I actually found experimentally that for I2V, applying only the low-noise LoRA to both experts at inference produced better results than using a separately trained high-noise LoRA + low-noise LoRA pair. The high-noise expert just didn't need its own LoRA — the reference image was already doing that work, and an overfit high-noise LoRA fought the image conditioning instead of working with it.
