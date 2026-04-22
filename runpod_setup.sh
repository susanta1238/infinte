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

# 7. Weights (~40 GB, one-time)
pip install "huggingface_hub[cli]"
mkdir -p weights
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk

echo ""
echo "Done. Start the API with:"
echo ""
echo "  python -u api.py \\"
echo "    --host 0.0.0.0 --port 8000 \\"
echo "    --ckpt-dir weights/Wan2.1-I2V-14B-480P \\"
echo "    --wav2vec-dir weights/chinese-wav2vec2-base \\"
echo "    --infinitetalk-dir weights/InfiniteTalk/single/infinitetalk.safetensors"
