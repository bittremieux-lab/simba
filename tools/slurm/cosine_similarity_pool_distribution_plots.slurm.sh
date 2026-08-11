#!/bin/bash
#SBATCH -J cosine_similarity_pool_distribution_plots
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/mces_pool_distribution_plots/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/mces_pool_distribution_plots/%x_%j.err

# Item 8a: does raw cosine similarity's "max" (over the pool) cluster close
# to 1, the way GT MCES's "min" clusters close to 0? Requires
# cosine_baseline_intermediates.slurm.sh to have completed first. CPU-only;
# uses a GPU node purely for fast, uncontended storage I/O.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/mces_pool_distribution_plots
COSINE_INTERMEDIATES_DIR=/sofia/projects/2026_053/simba_project/experiments/cosine_baseline_intermediates
CANDIDATES=/sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json
TEST_TO_TEST_PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_msg_exact_mces_1020

echo "===== Cosine similarity pool distribution plots (8a) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

cd "${SIMBA_DIR}"

uv run python tools/cosine_similarity_pool_distribution_plots.py \
    --cosine_intermediates_dir "${COSINE_INTERMEDIATES_DIR}" \
    --candidates "${CANDIDATES}" \
    --test_to_test_prepro_dir "${TEST_TO_TEST_PREPRO_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "===== Done: $(date) ====="
