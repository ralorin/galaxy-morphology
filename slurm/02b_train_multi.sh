#!/bin/bash
#SBATCH --job-name=gzm_train_multi
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Same work as 02_train_array.sh, packed differently: one job holding four GPUs and
# running four training processes side by side, each pinned to its own card. This
# is the honest way to ask gpu-large for four GPUs — every card is busy for the
# whole allocation — and it clears a long sweep in one go instead of queueing
# hundreds of single-GPU tasks.
#
#   FIRST=0 LAST=199 sbatch slurm/02b_train_multi.sh
#
# Worker k takes tasks FIRST+k, FIRST+k+4, FIRST+k+8, ... Runs that already have a
# metrics.json are skipped, so overlapping ranges do no harm.
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-galaxy}"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

JOBS=${GZM_JOBS:-${GZM_WORK:-$HOME/galaxy-morphology/work}/jobs/jobs.csv}
FIRST=${FIRST:-0}
LAST=${LAST:?set LAST to the highest task id, e.g. LAST=199}
N_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
# split the cores between the workers so the dataloaders do not fight
export GZM_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-16} / N_GPU ))

echo "GPUs asignadas: $CUDA_VISIBLE_DEVICES  ($N_GPU cards, $GZM_WORKERS loader workers each)"
echo "node: $(hostname)   tasks $FIRST..$LAST   start: $(date)"

for (( k=0; k<N_GPU; k++ )); do
  (
    export CUDA_VISIBLE_DEVICES=$k
    for (( t=FIRST+k; t<=LAST; t+=N_GPU )); do
      echo "[worker $k] task $t"
      python -m src.train --jobs-csv "$JOBS" --task-id "$t" || echo "[worker $k] task $t failed"
    done
  ) &
done
wait

echo "end: $(date)"
