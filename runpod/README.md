# Dimljus — RunPod Training

Quick-start guide for training Wan 2.2 LoRAs on RunPod.

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
    Videos/           ← your training clips (.mp4)
    Videos/*.txt      ← caption files (same name as video)
    Images/           ← reference images for I2V (optional)
```

### 4. Edit Dataset Config

Edit `/workspace/dimljus/runpod/dataset-config.toml`:
- Change the `video_directory` path to your dataset
- For I2V: uncomment the image subset section

## Training

Always run training inside tmux (so it survives if your browser disconnects):

```bash
cd /workspace/musubi-tuner
tmux new -s train
```

### T2V (Text-to-Video)

```bash
# Train high-noise expert
python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high

# Train low-noise expert
python /workspace/dimljus/runpod/train.py --variant t2v --noise_level low

# Train both sequentially
python /workspace/dimljus/runpod/train.py --variant t2v --noise_level both
```

### I2V (Image-to-Video)

```bash
python /workspace/dimljus/runpod/train.py --variant i2v --noise_level high
python /workspace/dimljus/runpod/train.py --variant i2v --noise_level low
```

### Custom Hyperparameters

```bash
python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high \
    --lr 5e-5 --rank 32 --alpha 32 --epochs 30
```

### Resume from Checkpoint

```bash
# Explicit path
python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high \
    --resume_from /workspace/outputs/my-lora-e25.safetensors

# Auto-detect: drop a .safetensors in /workspace/resume_checkpoints/
python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high
```

### Train with Speed LoRA (Lightning)

Bake a speed LoRA into the DiT before training. The resulting character LoRA
works together with the speed LoRA at inference for faster generation.

```bash
python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high \
    --merge lightning --merge_strength 0.8
```

## Download Results

Results are saved to `/workspace/outputs/`. Download via:
- Jupyter Lab file browser
- `scp -P PORT root@HOST:/workspace/outputs/*.safetensors .`

## After Pod Restart

Run setup again to reinstall Python packages (models stay cached):

```bash
bash /workspace/dimljus/runpod/setup.sh
```

## Default Hyperparameters

| Setting | High-Noise | Low-Noise |
|---------|-----------|-----------|
| Learning Rate | 1e-4 | 8e-5 |
| LoRA Rank | 16 | 24 |
| LoRA Alpha | 16 | 24 |
| Max Epochs | 30 | 50 |
| Save Every | 5 | 5 |

Shared: adamw8bit optimizer, polynomial scheduler, 0.01 weight decay, seed 42.

T2V: flow shift 3.0, boundary 875. I2V: flow shift 5.0, boundary 900.
