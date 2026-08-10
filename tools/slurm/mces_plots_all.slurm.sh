#!/bin/bash
#SBATCH -J mces_plots_all
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/mces_pool_distribution_plots/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/mces_pool_distribution_plots/%x_%j.err

# CPU-only plotting work (3e: per-spectrum MCES pool distribution /
# calibration / top-1 diagnostics) — requests a GPU node purely to land on a
# compute node with fast, uncontended storage I/O (the login-node filesystem
# has been extremely slow today); the scripts themselves never touch the
# GPU. Runs all three sequentially since each should finish in well under a
# minute once I/O isn't the bottleneck.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/mces_pool_distribution_plots
INTERMEDIATES_DIR=/sofia/projects/2026_053/simba_project/experiments/training/008_2_cosine_no_head_1gpu/retrieval_iceberg
GT_MCES_DIR=/sofia/projects/2026_053/simba_project/data/gt_mces_retrieval_candidates
TEST_TO_TEST_PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_msg_exact_mces_1020

echo "===== MCES plots (3e), all three scripts ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

cd "${SIMBA_DIR}"

echo "--- mces_pool_distribution_plots.py ---"
uv run python tools/mces_pool_distribution_plots.py \
    --intermediates_dir "${INTERMEDIATES_DIR}" \
    --gt_mces_dir "${GT_MCES_DIR}" \
    --test_to_test_prepro_dir "${TEST_TO_TEST_PREPRO_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "--- mces_calibration_plots.py ---"
uv run python tools/mces_calibration_plots.py \
    --intermediates_dir "${INTERMEDIATES_DIR}" \
    --gt_mces_dir "${GT_MCES_DIR}" \
    --test_to_test_prepro_dir "${TEST_TO_TEST_PREPRO_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "--- mces_top1_diagnostics.py ---"
uv run python tools/mces_top1_diagnostics.py \
    --intermediates_dir "${INTERMEDIATES_DIR}" \
    --gt_mces_dir "${GT_MCES_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "===== Done: $(date) ====="
