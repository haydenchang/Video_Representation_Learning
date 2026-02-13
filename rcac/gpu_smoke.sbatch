#!/bin/bash
#SBATCH -J gpu_smoke
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -p a30

set -euo pipefail

echo "== HOST =="
date
hostname

echo "== GPU =="
nvidia-smi

echo "== PYTHON =="
python --version

# Venv
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip wheel

# PyTorch CUDA wheel (use cu121 as a robust default wheel set)
python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "== TORCH CHECK =="
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1024,1024, device="cuda")
    y = x @ x
    print("matmul ok:", y.mean().item())
PY
