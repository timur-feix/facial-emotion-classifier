#!/bin/bash
#SBATCH --job-name=fer
#SBATCH --partition=gobi
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs checkpoints
source .venv/bin/activate

python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
EOF

python src/train.py --epochs 3 --batch-size 64 --num-workers 4 --lr 1e-3 --outdir checkpoints