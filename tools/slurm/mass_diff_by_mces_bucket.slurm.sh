#!/bin/bash
#SBATCH -J mass_diff_by_mces_bucket
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/mass_diff_by_mces_bucket/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/mass_diff_by_mces_bucket/%x_%j.err

# Diagnostic for calibrating a future mass-difference sampling weight (on
# top of the MCES-bucket resampling introduced in experiment 011): builds
# the real Gaetan-split-v2 training pair pool (same load_dataset +
# prepare_data path `simba train` uses) and plots the |mass_0 - mass_1|
# distribution per MCES sampling bucket. CPU-only; uses a GPU node purely
# for fast, uncontended storage I/O (same reasoning as
# tools/slurm/mces_plots_all.slurm.sh).

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
cd "${SIMBA_DIR}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

uv run python tools/mass_diff_by_mces_bucket.py

echo "===== Done: $(date) ====="
