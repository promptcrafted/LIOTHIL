"""Check if from_single_file maps weights to the correct model parameters."""
import torch
from safetensors.torch import load_file

# Load raw weights
sd = load_file("/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors")

# Load model via from_single_file
from diffusers.models import WanTransformer3DModel
model = WanTransformer3DModel.from_single_file(
    "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors",
    torch_dtype=torch.bfloat16,
    config="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    subfolder="transformer",
)
mp = dict(model.named_parameters())

# Expected key mapping (Wan native → Diffusers)
mappings = [
    ("blocks.0.self_attn.q.weight", "blocks.0.attn1.to_q.weight"),
    ("blocks.0.self_attn.k.weight", "blocks.0.attn1.to_k.weight"),
    ("blocks.0.cross_attn.q.weight", "blocks.0.attn2.to_q.weight"),
    ("blocks.0.cross_attn.k.weight", "blocks.0.attn2.to_k.weight"),
    ("blocks.0.ffn.0.weight", "blocks.0.ffn.net.0.proj.weight"),
    ("blocks.0.ffn.2.weight", "blocks.0.ffn.net.2.weight"),
    ("blocks.20.self_attn.q.weight", "blocks.20.attn1.to_q.weight"),
    ("blocks.39.cross_attn.v.weight", "blocks.39.attn2.to_v.weight"),
    ("patch_embedding.weight", "patch_embedding.weight"),
    ("proj_out.weight", "proj_out.weight"),
]

print("=== Weight mapping verification ===")
for raw_key, model_key in mappings:
    raw_w = sd.get(raw_key)
    mod_w = mp.get(model_key)
    if raw_w is None:
        print(f"  {raw_key} → NOT IN RAW FILE")
        continue
    if mod_w is None:
        print(f"  {raw_key} → {model_key} → NOT IN MODEL")
        continue
    match = torch.equal(raw_w.to(mod_w.dtype), mod_w)
    raw_mean = raw_w.float().mean().item()
    mod_mean = mod_w.float().mean().item()
    status = "MATCH" if match else "MISMATCH"
    print(f"  {status}: {raw_key}")
    print(f"          → {model_key}")
    print(f"          raw_mean={raw_mean:.6f}, model_mean={mod_mean:.6f}")

# Check: could attn1/attn2 be SWAPPED?
print("\n=== Swap check: is self_attn accidentally mapped to attn2? ===")
swap_pairs = [
    ("blocks.0.self_attn.q.weight", "blocks.0.attn2.to_q.weight"),
    ("blocks.0.cross_attn.q.weight", "blocks.0.attn1.to_q.weight"),
]
for raw_key, model_key in swap_pairs:
    raw_w = sd[raw_key]
    mod_w = mp[model_key]
    match = torch.equal(raw_w.to(mod_w.dtype), mod_w)
    status = "SWAPPED!" if match else "not swapped"
    print(f"  {status}: {raw_key} → {model_key}")

# Also check modulation / scale_shift_table
print("\n=== Special weights ===")
for raw_key in ["blocks.0.modulation", "scale_shift_table"]:
    if raw_key in sd:
        print(f"  Raw has: {raw_key} shape={list(sd[raw_key].shape)}")
for mod_key in ["blocks.0.scale_shift_table", "scale_shift_table"]:
    if mod_key in mp:
        print(f"  Model has: {mod_key} shape={list(mp[mod_key].shape)}")

print("\n=== FFN mapping check ===")
# Check if ffn.0 = ffn.net.0.proj (GELU activation in between)
for raw_key in ["blocks.0.ffn.0.weight", "blocks.0.ffn.0.bias",
                "blocks.0.ffn.2.weight", "blocks.0.ffn.2.bias"]:
    if raw_key in sd:
        print(f"  Raw: {raw_key} {list(sd[raw_key].shape)}")

for mod_key in sorted(mp.keys()):
    if "blocks.0.ffn" in mod_key:
        print(f"  Model: {mod_key} {list(mp[mod_key].shape)}")

del sd, model
print("\nDone.")
