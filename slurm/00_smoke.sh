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
#
# Deliberately no `set -e`: the point of a smoke test is to learn about every stage
# in one submission, not to stop at the first one that breaks. Failures are counted
# and listed at the end.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-galaxy}"
export PYTHONUNBUFFERED=1
# the nodes are shared; this keeps fragmentation from turning a tight
# fit into an out-of-memory error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SLURM_SUBMIT_DIR"
export GZM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

N=5000
SEED=99
FAILED=""

stage () {
    local label="$1"
    shift
    echo
    echo "=== $label ==="
    if "$@"; then
        echo "--- ok: $label"
    else
        echo "--- FAILED: $label"
        FAILED="$FAILED
  $label"
    fi
}

train () {
    python -m src.train --train-size "$N" --seed "$SEED" "$@"
}

echo "GPU asignada: $CUDA_VISIBLE_DEVICES"
echo "node: $(hostname)   start: $(date)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true

stage "pretrained convnet: the whole loop, and it keeps its weights" \
    train --arch resnet50 --label-mode soft --policy d4 --epochs 2 --save-checkpoint

stage "transformer: the timm attention path, weights kept for rollout" \
    train --arch vit_small --label-mode soft --policy d4 --epochs 1 --save-checkpoint

stage "swin: fixed input size and NHWC feature maps" \
    train --arch swin_tiny --label-mode soft --policy d4 --epochs 1

stage "scratch net at 128 px, hard labels" \
    train --arch cnn_small --label-mode hard --policy d4 --epochs 2

stage "orientation pooling: eight passes per step, reduced batch" \
    train --arch resnet50 --label-mode soft --policy none --epochs 1 --orientation-pooled

stage "the debiased target branch" \
    train --arch resnet50 --label-mode soft_debiased --policy d4 --epochs 1

RUN_CNN="resnet50_soft_d4_s224_full_bce_n${N}_seed${SEED}"
RUN_VIT="vit_small_soft_d4_s224_full_bce_n${N}_seed${SEED}"

stage "cross-survey scoring on Galaxy10 DECaLS" \
    python -m src.train --decals "$RUN_CNN"

stage "Grad-CAM on the convnet" \
    python -m src.xai --runs "$RUN_CNN" --n 6

stage "attention rollout on the transformer" \
    python -m src.xai --runs "$RUN_VIT" --n 6

echo
echo "end: $(date)"
if [ -z "$FAILED" ]; then
    echo "every stage passed; the sweep is safe to launch"
else
    echo "these stages failed:$FAILED"
    exit 1
fi
