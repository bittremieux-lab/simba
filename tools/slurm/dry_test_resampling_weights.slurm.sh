#!/bin/bash
#SBATCH -J dry_test_resampling_weights
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/dry_test_resampling_weights/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/dry_test_resampling_weights/%x_%j.err

# Dry-run diagnostic: simulates drawing from the REAL experiment-012-style
# weighted training sampler (same load_dataset + prepare_data path, same
# weights) to check whether the collapse seen in experiment 012 is explained
# by a few small groups (self-pairs, a fine bucket's low-mass-diff tier)
# getting repeated so many times within a fixed number of draws that they
# dominate the effective training signal despite a reasonable-looking
# aggregate bucket share. CPU-only; uses a GPU node purely for fast,
# uncontended storage I/O (same reasoning as tools/slurm/mces_plots_all.slurm.sh).

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
cd "${SIMBA_DIR}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

uv run python tools/dry_test_resampling_weights.py --n_draws 2000000

echo "===== Done: $(date) ====="
