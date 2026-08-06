#!/bin/bash
#SBATCH --job-name=gzm_smoke
#SBATCH --partition=gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Short runs over every code path that can break, before committing days of queue
# to the real sweep. Each one trains on 5,000 galaxies for one or two epochs, so
# the whole thing is well under an hour.
#
#   sbatch slurm/00_smoke.sh
#
# Seed 99 appears in no group of the experimental design, so these runs cannot
# collide with anything real. They still want deleting afterwards, because their
# train_size of 5,000 does appear in the learning-curve group and they would
# otherwise be averaged into it:
#
#   rm -rf $GZM_WORK/runs/*seed99* $GZM_WORK/results/xai/*seed99*
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-galaxy}"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
export GZM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

N=5000
SEED=99
COMMON="--train-size $N --seed $SEED --policy d4"

echo "GPU asignada: $CUDA_VISIBLE_DEVICES"
echo "node: $(hostname)   start: $(date)"

echo; echo "=== pretrained convnet: the whole loop, and it keeps its weights ==="
python -m src.train --arch resnet50 --label-mode soft $COMMON --epochs 2 --save-checkpoint

echo; echo "=== transformer: the timm attention path, weights kept for rollout ==="
python -m src.train --arch vit_small --label-mode soft $COMMON --epochs 1 --save-checkpoint

echo; echo "=== swin: fixed input size and NHWC feature maps ==="
python -m src.train --arch swin_tiny --label-mode soft $COMMON --epochs 1

echo; echo "=== scratch net at 128 px, hard labels ==="
python -m src.train --arch cnn_small --label-mode hard $COMMON --epochs 2

echo; echo "=== orientation pooling: eight passes per step, reduced batch ==="
python -m src.train --arch resnet50 --label-mode soft --train-size $N --seed $SEED \
    --policy none --epochs 1 --orientation-pooled

echo; echo "=== the debiased target branch ==="
python -m src.train --arch resnet50 --label-mode soft_debiased $COMMON --epochs 1

RUN_CNN=resnet50_soft_d4_s224_full_bce_n${N}_seed${SEED}
RUN_VIT=vit_small_soft_d4_s224_full_bce_n${N}_seed${SEED}

echo; echo "=== cross-survey scoring on Galaxy10 DECaLS ==="
python -m src.train --decals "$RUN_CNN"

echo; echo "=== Grad-CAM, and attention rollout on the transformer ==="
python -m src.xai --runs "$RUN_CNN" --n 6
python -m src.xai --runs "$RUN_VIT" --n 6

echo; echo "end: $(date)"
echo "if every stage above printed metrics, the sweep is safe to launch"
