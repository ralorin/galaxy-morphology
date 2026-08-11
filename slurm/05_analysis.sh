#!/bin/bash
#SBATCH --job-name=gzm_analysis
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Aggregation, the vote-model ceiling, the agreement-resolved tables, the
# bootstrap CIs and the significance tests, then the figures and the LaTeX. All
# pandas and scipy, so no GPU.
#
#   PAPER_DIR=~/paper5 sbatch slurm/05_analysis.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-galaxy}"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

PAPER=${PAPER_DIR:-${GZM_WORK:-$HOME/galaxy-morphology/work}/paper_assets}

echo "node: $(hostname)   start: $(date)"

python -m src.analysis
python -m src.stats
python -m src.figures --out "$PAPER"
python -m src.tables --out "$PAPER"
# and into the repository, so the results can be committed from here rather than
# copied off the cluster by hand
python -m src.publish

echo
echo "paper assets in $PAPER"
echo "end: $(date)"
