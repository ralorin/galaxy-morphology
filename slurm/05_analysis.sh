#!/bin/bash
#SBATCH --job-name=gzm_analysis
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# The first run of this took 3h38 against a four-hour wall clock and the second one
# went over it. The two stages that cost the time both read every run's per-galaxy
# predictions off disk, so it scales with the size of the sweep and there is no reason
# to keep shaving the limit. Twelve hours on a queue that allows two days.
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Aggregation, the vote-model ceiling, the agreement-resolved tables, the
# bootstrap CIs and the significance tests, then the figures and the LaTeX. All
# pandas and scipy, so no GPU.
#
#   PAPER_DIR=~/paper5 sbatch slurm/05_analysis.sh
#
# The last three stages take minutes and only read what the first two wrote, so if
# only a figure or a caption has changed there is no need to repeat the whole thing:
#
#   STAGES=assets PAPER_DIR=~/paper5 sbatch slurm/05_analysis.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-galaxy}"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

PAPER=${PAPER_DIR:-${GZM_WORK:-$HOME/galaxy-morphology/work}/paper_assets}
STAGES=${STAGES:-all}

echo "node: $(hostname)   start: $(date)   stages: $STAGES"

step () {
    echo
    echo "--- $* ---   $(date +%H:%M:%S)"
    "$@"
}

if [ "$STAGES" != "assets" ]; then
    step python -m src.analysis
    step python -m src.stats
fi

# The second dataset, when it is present. It is a separate sweep and a clone that
# has only run the Galaxy Zoo jobs should still get every other table, so a missing
# working set is a skip rather than a failure.
if [ -f "${GZM_WORK:-$HOME/galaxy-morphology/work}/arrays/fmh_table.csv" ]; then
    step python -m src.replication
else
    echo "no Fashion-MNIST-H working set, skipping the replication"
fi

step python -m src.figures --out "$PAPER"
step python -m src.tables --out "$PAPER"
# and into the repository, so the results can be committed from here rather than
# copied off the cluster by hand. --paper is explicit because the manuscript does not
# have to live inside the repository, and without it the figures and the LaTeX tables
# are silently left behind.
step python -m src.publish --paper "$PAPER"

echo
echo "paper assets in $PAPER"
echo "end: $(date)"
