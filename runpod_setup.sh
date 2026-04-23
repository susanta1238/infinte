#!/usr/bin/env bash
# Setup for RunPod image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# This image already provides:
#   - Python 3.11
#   - torch 2.4.0 built for CUDA 12.4
#   - nvcc, CUDA headers (devel variant) so flash-attn can build from source if needed
#
# Run once per pod, from /workspace (or wherever you want the repo).
set -euxo pipefail

# 0. System deps (ffmpeg + git-lfs are NOT in the base image)
apt-get update
apt-get install -y ffmpeg git git-lfs
git lfs install

# 1. Clone (skip if the repo is already mounted)
if [ ! -d InfiniteTalk ]; then
  git clone https://github.com/MeiGen-AI/InfiniteTalk.git
fi
cd InfiniteTalk

# 2. Sanity-check the pre-installed torch matches what we pin.
#    The base image ships torch 2.4.0+cu124 -- if that is still the case we use
#    it as-is and do NOT touch the torch install (avoids the 2.11/cu13
#    accidental-replace footgun).
python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print("FATAL: torch not importable in base image:", e); sys.exit(1)
print("base-image torch =", torch.__version__, "cuda_avail =", torch.cuda.is_available())
assert torch.__version__.startswith("2.4."), \
    f"unexpected torch in base image: {torch.__version__}"
PY

# 3. xformers (matching torch 2.4.0 + cu124 + py311).
pip install --no-deps xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124

# 4. flash-attn prerequisites, then flash-attn itself.
#    2.7.4.post1 publishes cp311 wheels for torch 2.4, so this is a fast wheel pull.
pip install psutil packaging wheel ninja
pip install flash_attn==2.7.4.post1 --no-build-isolation
pip install misaki[en]

# 5. Repo deps + API deps. The torch/xformers/flash_attn pins in requirements.txt
#    are identical to what we just installed, so pip sees them as satisfied and
#    does not touch them. accelerate is capped <1.3 to prevent it pulling torch 2.11.
pip install -r requirements.txt

# 6. Canary check: everything that matters must import AND cuda must work.
python - <<'PY'
import sys, torch, xformers, flash_attn
print("torch      =", torch.__version__)
print("xformers   =", xformers.__version__)
print("flash_attn =", flash_attn.__version__)
print("cuda avail =", torch.cuda.is_available())
print("device     =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.__version__.startswith("2.4."), f"torch got replaced: {torch.__version__}"
assert torch.cuda.is_available(), "CUDA not available"
PY

# 7. Weights — download only what the chosen profile needs.
#
# Profiles (set INFTALK_PROFILE before running this script):
#
#   single_lightx2v  Single-person + LightX2V LoRA, 4 steps    ~39 GB  *** FAST, A100/H100 default ***
#   single_fusionx   Single-person + FusionX LoRA, 8 steps     ~39 GB    fast, slightly better quality than lightx2v
#   single           Single-person, full precision, 40 steps   ~38 GB    best quality, slowest
#   single_fp8       Single-person, fp8 quantized              ~58 GB    half VRAM, ~40 steps
#   multi            Multi-person, full precision, 40 steps    ~38 GB
#   multi_fp8        Multi-person, fp8 quantized               ~58 GB
#   all              Full repo (single + multi + every quant)  ~197 GB
#
# Default is "single_lightx2v" — change with: INFTALK_PROFILE=single bash runpod_setup.sh
PROFILE="${INFTALK_PROFILE:-single_lightx2v}"

pip install "huggingface_hub[cli]"
mkdir -p weights

# Newer huggingface_hub (>=0.30) replaced `huggingface-cli` with `hf`.
# Prefer `hf` when present, fall back to the old name so this script works on either.
if command -v hf >/dev/null 2>&1; then
    HF="hf download"
else
    HF="huggingface-cli download"
fi
echo "using HuggingFace downloader: $HF"

# --- shared deps: needed for every profile ---
$HF Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir ./weights/Wan2.1-I2V-14B-480P
$HF TencentGameMate/chinese-wav2vec2-base \
    --local-dir ./weights/chinese-wav2vec2-base
$HF TencentGameMate/chinese-wav2vec2-base model.safetensors \
    --revision refs/pr/1 \
    --local-dir ./weights/chinese-wav2vec2-base

# --- profile-specific InfiniteTalk weights ---
case "$PROFILE" in
  single_lightx2v)
    # Single-person, full-precision base + LightX2V distilled LoRA (4 sampling steps).
    # Recommended for A100 / H100 / RTX 4090. ~4x faster than the 40-step default.
    $HF MeiGen-AI/InfiniteTalk \
        --include "single/*" \
        --local-dir ./weights/InfiniteTalk
    $HF Kijai/WanVideo_comfy \
        Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors \
        --local-dir ./weights
    INFTALK_FILE="weights/InfiniteTalk/single/infinitetalk.safetensors"
    EXTRA_FLAGS=$'--lora-dir weights/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors \\\n    --lora-scale 1.0 \\\n    --extra-args --sample_steps 4 --sample_text_guide_scale 1.0 --sample_audio_guide_scale 2.0 --sample_shift 2 --mode streaming --motion_frame 9'
    ;;
  single_fusionx)
    # Single-person, full-precision base + FusionX LoRA (8 sampling steps).
    # ~5x faster than 40-step, slightly higher quality than lightx2v.
    $HF MeiGen-AI/InfiniteTalk \
        --include "single/*" \
        --local-dir ./weights/InfiniteTalk
    $HF vrgamedevgirl84/Wan14BT2VFusioniX \
        Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
        --local-dir ./weights
    INFTALK_FILE="weights/InfiniteTalk/single/infinitetalk.safetensors"
    EXTRA_FLAGS=$'--lora-dir weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \\\n    --lora-scale 1.0 \\\n    --extra-args --sample_steps 8 --sample_text_guide_scale 1.0 --sample_audio_guide_scale 2.0 --sample_shift 2 --mode streaming --motion_frame 9'
    ;;
  single)
    $HF MeiGen-AI/InfiniteTalk \
        --include "single/*" \
        --local-dir ./weights/InfiniteTalk
    INFTALK_FILE="weights/InfiniteTalk/single/infinitetalk.safetensors"
    EXTRA_FLAGS=""
    ;;
  single_fp8)
    $HF MeiGen-AI/InfiniteTalk \
        --include "quant_models/infinitetalk_single_fp8.*" \
        --include "quant_models/t5_fp8.*" \
        --include "quant_models/t5_map_fp8.json" \
        --local-dir ./weights/InfiniteTalk
    INFTALK_FILE="weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors"
    EXTRA_FLAGS="--quant fp8 --quant-dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.json"
    ;;
  multi)
    $HF MeiGen-AI/InfiniteTalk \
        --include "multi/*" \
        --local-dir ./weights/InfiniteTalk
    INFTALK_FILE="weights/InfiniteTalk/multi/infinitetalk.safetensors"
    EXTRA_FLAGS=""
    ;;
  multi_fp8)
    $HF MeiGen-AI/InfiniteTalk \
        --include "quant_models/infinitetalk_multi_fp8.*" \
        --include "quant_models/t5_fp8.*" \
        --include "quant_models/t5_map_fp8.json" \
        --local-dir ./weights/InfiniteTalk
    INFTALK_FILE="weights/InfiniteTalk/quant_models/infinitetalk_multi_fp8.safetensors"
    EXTRA_FLAGS="--quant fp8 --quant-dir weights/InfiniteTalk/quant_models/infinitetalk_multi_fp8.json"
    ;;
  all)
    $HF MeiGen-AI/InfiniteTalk \
        --local-dir ./weights/InfiniteTalk
    INFTALK_FILE="weights/InfiniteTalk/single/infinitetalk.safetensors"
    EXTRA_FLAGS=""
    ;;
  *)
    echo "ERROR: unknown INFTALK_PROFILE='$PROFILE'."
    echo "Valid: single_lightx2v | single_fusionx | single | single_fp8 | multi | multi_fp8 | all"
    exit 1
    ;;
esac

echo ""
echo "Done. Profile: $PROFILE"
echo ""
echo "Start the API with:"
echo ""
echo "  python -u api.py \\"
echo "    --host 0.0.0.0 --port 8000 \\"
echo "    --ckpt-dir weights/Wan2.1-I2V-14B-480P \\"
echo "    --wav2vec-dir weights/chinese-wav2vec2-base \\"
echo "    --infinitetalk-dir $INFTALK_FILE \\"
if [ -n "$EXTRA_FLAGS" ]; then
echo "    $EXTRA_FLAGS"
fi
