#!/bin/bash
#SBATCH -J compute_val_cosine
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/compute_val_cosine/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/compute_val_cosine/%x_%j.err

# Computes raw spectral cosine similarity for every Gaetan-split-v2
# validation pair (classical, non-learned baseline vs. SIMBA's own
# predicted MCES) -- one-time, experiment-independent artifact, reused by
# every experiment sharing this val set (see tools/compute_val_cosine.py's
# module docstring). CPU-only; uses a GPU node purely for fast,
# uncontended storage I/O (same reasoning as tools/slurm/mces_plots_all.slurm.sh).

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
cd "${SIMBA_DIR}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

uv run python tools/compute_val_cosine.py

echo "===== Done: $(date) ====="
