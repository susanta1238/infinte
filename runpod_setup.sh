#!/usr/bin/env bash
# Setup for RunPod (Ubuntu 22.04 + CUDA 12.1, 24GB+ VRAM GPU)
# Run once per pod.
set -euxo pipefail

# 0. System deps
apt-get update
apt-get install -y ffmpeg git git-lfs
git lfs install

# 1. Clone (skip if the repo is already mounted)
if [ ! -d InfiniteTalk ]; then
  git clone https://github.com/MeiGen-AI/InfiniteTalk.git
fi
cd InfiniteTalk

# 2. Conda env (RunPod base images usually have miniconda at /opt/conda)
source /opt/conda/etc/profile.d/conda.sh || true
conda create -y -n multitalk python=3.10
conda activate multitalk

# 3. PyTorch + CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# 4. flash-attn (wheel build takes a while)
pip install misaki[en]
pip install psutil packaging wheel ninja
pip install flash_attn==2.7.4.post1

# 5. Repo deps + API deps
pip install -r requirements.txt
pip install -r api_requirements.txt

# 6. Weights (~40 GB, one-time)
pip install "huggingface_hub[cli]"
mkdir -p weights
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk

echo ""
echo "Done. Start the API with:"
echo ""
echo "  python api.py \\"
echo "    --host 0.0.0.0 --port 8000 \\"
echo "    --ckpt-dir weights/Wan2.1-I2V-14B-480P \\"
echo "    --wav2vec-dir weights/chinese-wav2vec2-base \\"
echo "    --infinitetalk-dir weights/InfiniteTalk/single/infinitetalk.safetensors"
