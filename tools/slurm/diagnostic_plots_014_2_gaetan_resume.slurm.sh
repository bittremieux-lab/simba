#!/bin/bash
#SBATCH -J simba_diag_plots_014_2_resume
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots/%x_%j.err

# Resume of diagnostic_plots_014_2_gaetan_1gpu.slurm.sh from step 4 onward:
# the first run's steps 1-3 and 5a/5b already completed and produced valid
# output (embeddings/intermediates don't depend on the bug fixed below, and
# mces_calibration_plots.py/mces_top1_diagnostics.py never touch
# build_retrieval_comparison_table.py's CSV at all). Step 5c crashed on a
# genuine pre-existing schema bug in build_retrieval_comparison_table.py
# (never wrote test_precursor_mz/candidate_precursor_mz, which
# plot_retrieval_comparison_checks.py has always expected to read) -- fixed
# in build_retrieval_comparison_table.py, not something these Gaetan-specific
# changes introduced. CPU-only (no GPU needed for any of these 3 steps).

set -euo pipefail
module load uv
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
cd "${SIMBA_DIR}"

echo "===== 014_2 Gaetan diagnostic-plot pipeline (resume from step 4) ====="
echo "Job ID: $SLURM_JOB_ID"
date

MGF=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test.mgf
CANDIDATES=/sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json
CANDIDATE_TSV_1=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_existing_overlap.tsv
CANDIDATE_TSV_2=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_new.tsv
ICEBERG_PREDS_1=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_test_official/preds.hdf5
ICEBERG_PREDS_2=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_gaetan_test_new/preds.hdf5

SIMBA_INTER=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_014_2_regression
COSINE_INTER=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/cosine_baseline_intermediates_gaetan_test
COSINE_TSV=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/cosine_iceberg/retrieval_results.tsv
GT_MCES_DIR=/sofia/projects/2026_053/simba_project/data/gt_mces_gaetan_test
TABLE_CSV=/sofia/projects/2026_053/simba_project/experiments/retrieval_comparison_table/retrieval_comparison_table_014_2_gaetan.csv
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots

echo ""
echo "--- Step 4/5: build retrieval comparison table (014_2 vs cosine, Gaetan) ---"
uv run python tools/build_retrieval_comparison_table.py \
    --simba_intermediates_dir "${SIMBA_INTER}" \
    --cosine_intermediates_dir "${COSINE_INTER}" \
    --candidates "${CANDIDATES}" \
    --gt_mces_dir "${GT_MCES_DIR}" \
    --mgf "${MGF}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --output_csv "${TABLE_CSV}"

echo ""
echo "--- Step 5c/5: plot_retrieval_comparison_checks.py ---"
uv run python tools/plot_retrieval_comparison_checks.py \
    --table_csv "${TABLE_CSV}" \
    --simba_retrieval_results_tsv "${SIMBA_INTER}/retrieval_results.tsv" \
    --cosine_retrieval_results_tsv "${COSINE_TSV}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "--- Step 5d/5: plot_confusion_matrix_examples.py ---"
uv run python tools/plot_confusion_matrix_examples.py \
    --table_csv "${TABLE_CSV}" \
    --mgf "${MGF}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "===== Pipeline complete: $(date) ====="
ls -la "${OUTPUT_DIR}"
