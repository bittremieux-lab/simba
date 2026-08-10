#!/bin/bash
#SBATCH -J ood_generalization_check
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/mces_pool_distribution_plots/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/mces_pool_distribution_plots/%x_%j.err

# CPU-only (3e: MAE/Spearman summary, per-spectrum query, post-bugfix) —
# requests a GPU node purely for fast, uncontended storage I/O; the script
# itself never touches the GPU.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== ood_generalization_check.py ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

cd "${SIMBA_DIR}"

uv run python tools/ood_generalization_check.py \
    --intermediates_dir /sofia/projects/2026_053/simba_project/experiments/training/008_2_cosine_no_head_1gpu/retrieval_iceberg \
    --gt_mces_dir /sofia/projects/2026_053/simba_project/data/gt_mces_retrieval_candidates \
    --test_to_test_prepro_dir /sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_msg_exact_mces_1020

echo "===== Done: $(date) ====="
