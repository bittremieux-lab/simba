#!/bin/bash
#SBATCH -J simba_val_vs_tt_014_2
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots/%x_%j.err

set -euo pipefail
module load uv
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
cd "${SIMBA_DIR}"

echo "===== val vs test-to-test binned-box comparison (014_2) ====="
date

uv run python tools/plot_val_vs_test_to_test_binned_box.py \
    --val_parquet /sofia/projects/2026_053/simba_project/experiments/training/014_2_mces_bucket_mlp_1gpu/val_pairs_val_consolidated.parquet \
    --val_step 229000 \
    --test_intermediates_dir /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_014_2_regression \
    --test_to_test_prepro_dir /sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2 \
    --output_dir /sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots

echo "===== Done: $(date) ====="
