#!/bin/bash
#SBATCH -J simba_diag_plots_014_2
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots/%x_%j.err

# Reproduces the 6-plot diagnostic pipeline (originally built for checkpoint
# 008_2 on the official split -- NOTES_GT_MCES_RETRIEVAL.md item 3e) for
# checkpoint 014_2 on the Gaetan split. All 6 PNGs land in this job's
# --output_dir (experiments/014_2_gaetan_diagnostic_plots/), per request.
#
# GT-MCES ground truth for Gaetan test-to-candidate pairs doesn't exist
# anywhere (the asimov2-computed data/gt_mces_retrieval_candidates/ only
# covers the official split) -- built here instead from the two pre-existing
# all-vs-all matrices (lb_matrix.npy + all_smiles_mces.hdf5, same ones
# oracle_retrieval_gt_mces.py already uses for the Oracle GT-MCES-NN row),
# no asimov2 needed. lb_matrix.npy is loaded fully into RAM (~116GB), not
# memory-mapped (mmap'ing it turns ~80M scattered reads into a multi-hour
# job on this filesystem -- see oracle_retrieval_gt_mces.py's own module
# docstring) -- this cluster assigns memory per CPU core automatically (no
# explicit --mem flag), so --cpus-per-task=24 here (~7.8GB/core on this
# partition) gives ~187GB, comfortably above what's needed.
#
# This run is REGRESSION-ONLY (head_mode=cosine_no_head's cosine-similarity
# score, no --corn_corrected) -- none of these 5 scripts have any CORN
# bucket-head awareness (verified by direct code reading), and adding that
# would mean a genuine pairwise bucket-head pass over every scored pair
# (hundreds of thousands, not just each query's top-1), a substantially
# bigger undertaking than this run. Flagged as follow-up work, not attempted
# here.
#
# candidate_tsv/iceberg_preds are split across two files for the Gaetan
# split (existing_overlap + new delta) -- three of these five scripts
# (cosine_baseline_intermediates.py, build_retrieval_comparison_table.py,
# plot_confusion_matrix_examples.py) only accepted a single path each before
# this run; patched to accept nargs="+" the same way simba_retrieval_iceberg.py
# already did, reusing load_candidate_index/load_iceberg_spectra's existing
# list support rather than duplicating the merge logic.

set -euo pipefail
module load uv
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
cd "${SIMBA_DIR}"

echo "===== 014_2 Gaetan diagnostic-plot pipeline ====="
echo "Job ID: $SLURM_JOB_ID"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CHECKPOINT=/sofia/projects/2026_053/simba_project/experiments/training/014_2_mces_bucket_mlp_1gpu/checkpoint-epoch=22-step=229000.ckpt
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
GAETAN_PREPRO=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
TABLE_CSV=/sofia/projects/2026_053/simba_project/experiments/retrieval_comparison_table/retrieval_comparison_table_014_2_gaetan.csv
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots

echo ""
echo "--- Step 1/5: SIMBA retrieval + intermediates (014_2, regression) ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${CHECKPOINT}" \
    --head_mode cosine_no_head \
    --precursor_mass_mode theoretical \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test --batch_size 512 --skip_mces \
    --intermediates_dir "${SIMBA_INTER}" \
    --output_tsv "${SIMBA_INTER}/retrieval_results.tsv"

echo ""
echo "--- Step 2/5: cosine-baseline intermediates (Gaetan test) ---"
uv run python tools/cosine_baseline_intermediates.py \
    --mgf "${MGF}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --intermediates_dir "${COSINE_INTER}" \
    --split test

echo ""
echo "--- Step 3/5: GT-MCES lookup for Gaetan test-to-candidate pairs (max(lb_matrix, hdf5), no asimov2) ---"
uv run python tools/build_gt_mces_from_matrices.py \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --split test \
    --lb_matrix /sofia/projects/2026_053/simba_project/data/massspecgym/lb_matrix.npy \
    --lb_smiles /sofia/projects/2026_053/simba_project/data/massspecgym/lb_matrix.smiles.txt \
    --hdf5_mces_path /sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/all_smiles_mces.hdf5 \
    --output_dir "${GT_MCES_DIR}"

echo ""
echo "--- Step 4/5: build retrieval comparison table (014_2 vs cosine, Gaetan) ---"
mkdir -p "$(dirname "${TABLE_CSV}")"
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
echo "--- Step 5a/5: mces_calibration_plots.py (binned-box x2) ---"
uv run python tools/mces_calibration_plots.py \
    --intermediates_dir "${SIMBA_INTER}" \
    --gt_mces_dir "${GT_MCES_DIR}" \
    --test_to_test_prepro_dir "${GAETAN_PREPRO}" \
    --output_dir "${OUTPUT_DIR}" \
    --force_recompute

echo ""
echo "--- Step 5b/5: mces_top1_diagnostics.py ---"
uv run python tools/mces_top1_diagnostics.py \
    --intermediates_dir "${SIMBA_INTER}" \
    --gt_mces_dir "${GT_MCES_DIR}" \
    --output_dir "${OUTPUT_DIR}"

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
