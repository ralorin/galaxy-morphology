#!/bin/bash
#SBATCH --job-name=gzm_xai
#SBATCH --partition=gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Grad-CAM for the convnets, attention rollout for the transformers, plus the
# deletion, insertion and background-reliance scores. The deletion and insertion
# curves are 21 forward passes per galaxy per model, which is why this wants a GPU
# even though nothing is being trained.
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-galaxy}"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

N=${GZM_XAI_N:-48}

echo "GPU asignada: $CUDA_VISIBLE_DEVICES"
echo "node: $(hostname)   start: $(date)"

python -m src.xai --all-checkpoints --n "$N"

echo "end: $(date)"
