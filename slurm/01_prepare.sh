#!/bin/bash
#SBATCH --job-name=gzm_prepare
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Decodes the ~240k Galaxy Zoo 2 jpegs into a uint8 memmap and builds the label
# table and the splits, then does the same for Galaxy10 DECaLS. No GPU involved,
# so this belongs on the cpu partition.
#
# The downloads themselves are NOT here: compute nodes have no outbound network.
# Run `python -m src.download_data` on the login node first.
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-galaxy}"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

# one decoder process per allocated core
export GZM_WORKERS=${SLURM_CPUS_PER_TASK:-16}

echo "node: $(hostname)   start: $(date)"
python config.py

python -m src.prepare_gz2 --stage all
python -m src.prepare_decals

echo "end: $(date)"
