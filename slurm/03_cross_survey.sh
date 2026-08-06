#!/bin/bash
#SBATCH --job-name=gzm_cross
#SBATCH --partition=gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Scores every run that kept its weights on Galaxy10 DECaLS. Inference only, one
# GPU, a few minutes per checkpoint.
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-galaxy}"
export PYTHONUNBUFFERED=1
# the nodes are shared; this keeps fragmentation from turning a tight
# fit into an out-of-memory error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SLURM_SUBMIT_DIR"

export GZM_WORKERS=${SLURM_CPUS_PER_TASK:-8}
RUNS_DIR=${GZM_WORK:-$HOME/galaxy-morphology/work}/runs

echo "GPU asignada: $CUDA_VISIBLE_DEVICES"
echo "node: $(hostname)   start: $(date)"

for ckpt in "$RUNS_DIR"/*/checkpoint.pt; do
    [ -e "$ckpt" ] || { echo "no checkpoints found in $RUNS_DIR"; exit 1; }
    run_id=$(basename "$(dirname "$ckpt")")
    echo "--- $run_id"
    python -m src.train --decals "$run_id"
done

echo "end: $(date)"
