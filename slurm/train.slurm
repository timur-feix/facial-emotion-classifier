#!/bin/bash
#SBATCH --job-name=fer
#SBATCH --partition=NvidiaAll
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO_DIR="$HOME/facial-emotion-classifier"
BRANCH="custom-model-gpu-access"

cd "$REPO_DIR"

echo "=== Git update ==="
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "Running commit:"
git rev-parse HEAD
echo "=================="

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs checkpoints
source .venv/bin/activate


nvidia-smi

python src/train.py --epochs 10 --batch-size 64 --num-workers 4 --lr 1e-3 --outdir checkpoints