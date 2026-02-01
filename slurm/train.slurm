#!/bin/bash
#SBATCH --job-name=fer
#SBATCH --partition=Krater
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs checkpoints
source .venv/bin/activate


nvidia-smi

python src/train.py --epochs 20 --batch-size 64 --num-workers 4 --lr 1e-3 --outdir checkpoints --debug True