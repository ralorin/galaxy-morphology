#!/bin/bash
#SBATCH --job-name=gzm_replication
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# The same pipeline on a second, non-astronomical dataset: Fashion-MNIST-H, which
# carries about sixty-seven annotations per test image. The point is to show that
# the label ceiling and the concentration of error on contested items are properties
# of panel annotation rather than of galaxies.
#
#   sbatch slurm/06_replication.sh
#
# Twelve thousand 28-pixel images per run, so this is small: the whole grid is
# minutes per run rather than hours. The four-hour wall clock is slack, not need.

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-galaxy}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SLURM_SUBMIT_DIR"

PAIR=${PAIR:-"pullover coat"}
SEEDS=${SEEDS:-"0 1 2"}
ARCHS=${ARCHS:-"resnet50 convnext_tiny vit_small"}

echo "node: $(hostname)   start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "--- preparing Fashion-MNIST-H: $PAIR ---"
python -m src.prepare_fmh --pair $PAIR

for arch in $ARCHS; do
    for seed in $SEEDS; do
        echo
        echo "--- $arch seed $seed ---   $(date +%H:%M:%S)"
        python -m src.train --dataset fmh --arch "$arch" --label-mode hard \
               --policy flip --seed "$seed" --epochs 12 --patience 4
    done
done

echo
echo "end: $(date)"
