# Dimljus — RunPod Training

Quick-start guide for training Wan 2.2 LoRAs on RunPod using dimljus natively.

## Setup

### 1. Create a RunPod Pod

- **GPU**: H100 80GB (recommended) or A100 80GB
- **Template**: RunPod PyTorch 2.x
- **Container Disk**: **50 GB** (default 20 GB will run out)
- **Volume Disk**: 200 GB (models + datasets + outputs)
- **Environment Variables**: Set `HF_TOKEN` to your HuggingFace token

### 2. Clone and Setup

Open a terminal in Jupyter Lab:

```bash
# Clone the repo
cd /workspace
git clone https://github.com/alvdansen/dimljus.git

# Run setup (installs packages, downloads ~35GB of models)
bash /workspace/dimljus/runpod/setup.sh
```

### 3. Upload Your Dataset

Via Jupyter Lab file browser, upload to:

```
/workspace/datasets/my_dataset/
    clip_001.mp4          ← training video clips
    clip_001.txt          ← caption sidecar (same stem as video)
    clip_002.mp4
    clip_002.txt
    ...
```

You also need a `dimljus_data.yaml` in the dataset directory. See the
[data config docs](../docs/) for the schema, or use a minimal one:

```yaml
# /workspace/datasets/my_dataset/dimljus_data.yaml
video:
  target_fps: 16
  min_duration: 1.0
```

### 4. Create a Training Config

```bash
cp /workspace/dimljus/examples/full_train.yaml /workspace/my_train.yaml
```

Edit `/workspace/my_train.yaml`:
- Set `data_config` to point to your dataset's `dimljus_data.yaml`
- Adjust epochs, learning rates, and output paths as needed
- Model file paths are already set for RunPod (`/workspace/models/...`)

See `runpod/test-train.yaml` for a minimal example.

## Training

Always run training inside tmux (survives browser disconnects):

```bash
tmux new -s train
```

### Full Run (Encode + Train)

```bash
python /workspace/dimljus/runpod/train.py --config /workspace/my_train.yaml
```

This runs all three steps automatically:
1. **Cache latents** — encode videos through VAE
2. **Cache text** — encode captions through T5
3. **Train** — run the dimljus training loop

### Dry Run (Validate Config)

```bash
python /workspace/dimljus/runpod/train.py --config /workspace/my_train.yaml --dry-run
```

Validates your config and prints the training plan without using the GPU.

### Encode Only

```bash
python /workspace/dimljus/runpod/train.py --config /workspace/my_train.yaml --encode-only
```

Build latent and text caches without starting training. Useful for
verifying encoding works before committing to a long training run.

### Skip Encoding

```bash
python /workspace/dimljus/runpod/train.py --config /workspace/my_train.yaml --skip-encoding
```

Skip encoding steps (use existing caches). Useful when re-running
training with different hyperparameters on the same dataset.

## Download Results

Results are saved to `/workspace/outputs/` (or wherever `save.output_dir` points).
Download via:
- Jupyter Lab file browser
- `scp -P PORT root@HOST:/workspace/outputs/*.safetensors .`

## After Pod Restart

Run setup again to reinstall Python packages (models stay cached on `/workspace`):

```bash
bash /workspace/dimljus/runpod/setup.sh
```

## Default Hyperparameters

All training configuration lives in the YAML config file. See
`examples/full_train.yaml` for the full reference with all options documented.

Key defaults for fork-and-specialize MoE training:

| Setting | Unified | High-Noise Expert | Low-Noise Expert |
|---------|---------|-------------------|------------------|
| Learning Rate | 5e-5 | 1e-4 | 8e-5 |
| Epochs | 15 | 30 | 50 |
| LoRA Rank | 16 | 16 | 16 |
| LoRA Alpha | 16 | 16 | 16 |

Shared: adamw8bit optimizer, cosine_with_min_lr scheduler, 0.01 weight decay, seed 42.
