#!/bin/bash
# =============================================================================
# RunPod Setup Script for Dimljus Training
# =============================================================================
# Run this ONCE when you first create your pod, AND again after every restart
# (pod restarts wipe the container disk where Python packages live).
#
# Everything important lives on /workspace which persists across restarts.
# Python packages need reinstalling after restart, but models don't.
#
# IMPORTANT: Set Container Disk to 50 GB when creating the pod.
# The default 20 GB will run out during model downloads.
#
# Usage:
#   bash /workspace/dimljus/runpod/setup.sh
# =============================================================================

set -e  # Stop on any error

echo "=============================================="
echo "  Dimljus — RunPod Setup"
echo "=============================================="

# --- 1. System packages ---
echo ""
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq ffmpeg python3-opencv tmux > /dev/null 2>&1
echo "  Done: ffmpeg, opencv, tmux"

# --- 2. Clone and install dimljus ---
echo ""
echo "[2/5] Setting up dimljus..."
if [ -d "/workspace/dimljus" ]; then
    echo "  dimljus already exists — pulling latest..."
    cd /workspace/dimljus
    git pull
else
    echo "  Cloning dimljus..."
    cd /workspace
    git clone https://github.com/alvdansen/dimljus.git
    cd /workspace/dimljus
fi

# Install dimljus with Wan training dependencies.
# NOTE: These live on the container disk, so they need reinstalling after restart.
pip install -q -e ".[wan]" 2>/dev/null || pip install -q -e . 2>/dev/null
pip install -q huggingface_hub hf_transfer
echo "  Done: dimljus installed"

# --- 3. Set up HuggingFace ---
echo ""
echo "[3/5] Configuring HuggingFace..."
export HF_HUB_ENABLE_HF_TRANSFER=1

# Check if HF_TOKEN is set (should be in pod environment variables)
if [ -z "$HF_TOKEN" ]; then
    echo "  WARNING: HF_TOKEN environment variable not set!"
    echo "  Model downloads may fail for gated repos."
    echo "  Fix: Add HF_TOKEN in RunPod pod settings > Environment Variables"
    echo "  Or run: huggingface-cli login"
else
    echo "  HF_TOKEN found"
    # Log in so downloads work for gated repos
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
fi

# --- 4. Download models ---
echo ""
echo "[4/5] Downloading models to /workspace/models/..."
echo "  This takes 10-20 minutes on first run (downloading ~35GB)."
echo "  Subsequent runs skip already-downloaded files."

mkdir -p /workspace/models

# Helper function: download if not already present.
# Downloads to /workspace (volume disk) not /tmp (container disk) to avoid
# "no space left on device" errors. Clears HF cache after each download.
download_model() {
    local repo=$1
    local filename=$2
    local dest=$3

    if [ -f "$dest" ]; then
        echo "  [CACHED] $(basename $dest)"
        return
    fi

    echo "  [DOWNLOADING] $repo / $(basename $filename) ..."
    huggingface-cli download "$repo" "$filename" \
        --local-dir /workspace/models/hf_tmp \
        --quiet 2>/dev/null
    mv "/workspace/models/hf_tmp/$filename" "$dest"
    rm -rf /workspace/models/hf_tmp
    # Clear HF cache on container disk to prevent filling it up
    rm -rf /root/.cache/huggingface/hub/* /tmp/hf_* 2>/dev/null || true
    echo "  Done: $(basename $dest)"
}

# Wan 2.2 T2V DiT weights (high + low noise experts)
download_model \
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors" \
    "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"

download_model \
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors" \
    "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors"

# Wan 2.2 I2V DiT weights (high + low noise experts)
download_model \
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
    "/workspace/models/wan2.2_i2v_high_noise_14B_fp16.safetensors"

download_model \
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
    "/workspace/models/wan2.2_i2v_low_noise_14B_fp16.safetensors"

# VAE (shared across all Wan models)
download_model \
    "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
    "split_files/vae/wan_2.1_vae.safetensors" \
    "/workspace/models/wan_2.1_vae.safetensors"

# T5 text encoder (shared across all Wan models)
download_model \
    "Wan-AI/Wan2.1-I2V-14B-720P" \
    "models_t5_umt5-xxl-enc-bf16.pth" \
    "/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"

# --- 5. Create directory structure ---
echo ""
echo "[5/5] Creating directory structure..."
mkdir -p /workspace/datasets
mkdir -p /workspace/outputs
echo "  Done"

# --- Done ---
echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "  Models downloaded:"
ls -lh /workspace/models/*.safetensors /workspace/models/*.pth 2>/dev/null | awk '{print "    " $5 "  " $9}'
echo ""
echo "  Next steps:"
echo "  1. Upload your dataset via Jupyter Lab"
echo "     Drag your video clips + .txt captions into /workspace/datasets/my_dataset/"
echo ""
echo "  2. Copy and edit a training config:"
echo "     cp /workspace/dimljus/examples/full_train.yaml /workspace/my_train.yaml"
echo "     Edit data_config to point to your dataset directory"
echo ""
echo "  3. Start training:"
echo "     tmux new -s train"
echo "     python /workspace/dimljus/runpod/train.py --config /workspace/my_train.yaml"
echo ""
echo "  4. Download results from /workspace/outputs/ via Jupyter Lab"
echo ""
echo "  AFTER POD RESTART: Run this script again to reinstall Python packages."
echo "  Models and dataset won't re-download (they're on /workspace)."
echo ""
