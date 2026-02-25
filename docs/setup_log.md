# Dimljus Environment Setup & Model Download Log

**Date:** 2026-02-22

## Step 1: Python Installation
- Machine had Python 3.14.0rc3 (release candidate — not compatible with PyTorch)
- Installed **Python 3.12.10** via `winget install Python.Python.3.12 --source winget`
- Verified: `py -3.12 --version` → Python 3.12.10

## Step 2: Virtual Environment
- Created: `py -3.12 -m venv C:\Users\minta\Projects\dimljus\.venv`
- Activate: `source /c/Users/minta/Projects/dimljus/.venv/Scripts/activate` (bash)
- Or Windows CMD: `C:\Users\minta\Projects\dimljus\.venv\Scripts\activate`

## Step 3: Dependencies Installed
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install safetensors huggingface-hub
```

Versions installed:
- torch 2.10.0+cpu (CPU-only — sufficient for weight analysis, no GPU needed)
- safetensors 0.7.0
- huggingface_hub 1.4.1
- pyyaml 6.0.3 (pulled in as dependency — also needed for Dimljus configs)

Note: numpy warning on torch import is harmless (not needed for this work).

## Step 4: Model Downloads

### T2V Model
- **Repo:** `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- **Command:** `hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir C:\Users\minta\Projects\dimljus\models\Wan2.2-T2V-A14B-Diffusers`
- **Local path:** `C:\Users\minta\Projects\dimljus\models\Wan2.2-T2V-A14B-Diffusers`
- **Contents:** 49 files — transformer/ (expert 1), transformer_2/ (expert 2), text_encoder/, vae/, tokenizer/, scheduler/
- **Status:** Downloading (~66 GB of ~126 GB as of 13:59)

### I2V Model
- **Repo:** `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- **Command:** `hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers --local-dir C:\Users\minta\Projects\dimljus\models\Wan2.2-I2V-A14B-Diffusers`
- **Local path:** `C:\Users\minta\Projects\dimljus\models\Wan2.2-I2V-A14B-Diffusers`
- **Contents:** 50 files (1 extra vs T2V — likely image processor/encoder for reference image)
- **Status:** Downloading (~43 GB as of 13:59)
- **Note:** First attempt failed — incorrect repo name `Wan-AI/Wan2.2-I2V-14B-720P-Diffusers`. Correct name has no "720P".

### Download Notes
- Downloads run unauthenticated (no HF_TOKEN set) — slower rate limits
- Effective speed: ~30-50 MB/s per download
- Both downloading in parallel
- To speed up future downloads: `hf auth login` to set a token

## Step 5: Analysis Script
- **Written:** `C:\Users\minta\Projects\dimljus-kit\tools\analyze_experts.py`
- **Syntax verified:** OK
- **Supports three modes:**
  1. `--model-dir` — compare experts within a single model (transformer/ vs transformer_2/)
  2. `--dir-a` / `--dir-b` — compare any two arbitrary transformer directories
  3. `--batch` — run all four T2V/I2V cross-comparisons at once

### Usage (after downloads complete)
```bash
# Activate the environment
source /c/Users/minta/Projects/dimljus/.venv/Scripts/activate

# Single model comparison
python C:/Users/minta/Projects/dimljus-kit/tools/analyze_experts.py \
  --model-dir C:/Users/minta/Projects/dimljus/models/Wan2.2-T2V-A14B-Diffusers

# All four comparisons at once
python C:/Users/minta/Projects/dimljus-kit/tools/analyze_experts.py \
  --batch \
  --t2v-dir C:/Users/minta/Projects/dimljus/models/Wan2.2-T2V-A14B-Diffusers \
  --i2v-dir C:/Users/minta/Projects/dimljus/models/Wan2.2-I2V-A14B-Diffusers
```

### The Four Comparisons
1. **T2V Expert 1 vs T2V Expert 2** — how much did T2V experts diverge from each other
2. **I2V Expert 1 vs I2V Expert 2** — how much did I2V experts diverge from each other
3. **T2V Expert 1 vs I2V Expert 1** — high-noise expert: how did I2V's reference image change it
4. **T2V Expert 2 vs I2V Expert 2** — low-noise expert: how did I2V's reference image change it

### What We Expect to Learn
- Which layer types diverged most (attention? FFN? modulation? norms?)
- Which block positions diverged most (early? late? uniform?)
- Overall similarity metric (confirming or refuting the ~90% research estimate)
- Whether I2V experts diverged differently than T2V experts
- Shape mismatches between T2V and I2V (revealing I2V's image conditioning architecture)
- All of this directly informs LoRA rank allocation and the fork-and-specialize training strategy

## Download Progress Log
| Time  | T2V Size | I2V Size |
|-------|----------|----------|
| 13:34 | 33 GB    | 4.1 GB   |
| 13:43 | 35 GB    | 7.9 GB   |
| 13:48 | 44 GB    | 23 GB    |
| 13:53 | 58 GB    | 37 GB    |
| 13:59 | 66 GB    | 43 GB    |
