#!/bin/bash
#SBATCH -J simba_retrieval_iceberg_logloss_a40
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_logloss_a40/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_logloss_a40/%x_%j.err

# SIMBA+ICEBERG retrieval for the log-loss a40 checkpoint (see
# NOTES_014_2_LOGLOSS.md) -- same architecture as 014_2 (head_mode=
# cosine_no_head, mces_bucket.use_mlp=true), only difference is the training
# loss (model.tasks.mces.use_log_loss=true, log_loss_a=40, i.e. trained
# toward log(MCES+1) instead of plain MSE). Directly tests Wout's point
# (Slack 2026-08-28): "retrieving the best candidate when all candidates
# share mass+formula feels like a different task from open-database analog
# retrieval" -- this IS that different (ICEBERG, bounded-candidate-pool)
# task, run here to see whether the log-loss's better bucket-balanced-
# accuracy (see NOTES_014_2_LOGLOSS.md results) translates into better
# Hit@k here too, not just a proxy metric.
#
# Mirrors NOTES_RETRIEVAL_014_2_CORN.md's pipeline step 4 exactly (same
# Gaetan-test ICEBERG candidate files, same --precursor_mass_mode
# theoretical, same --min_peaks 6) -- runs BOTH raw-regression and
# CORN-corrected ranking in one job (sequential, same loaded embeddings
# reused isn't possible across the two separate script invocations, but
# each run is only ~a few minutes). Reuses the ORIGINAL v1 Gaetan-test
# ICEBERG candidate files -- confirmed byte-identical mapping.pkl between
# v1/v2, so no rebuild needed. Comparable rows already exist for 014_2
# itself, ICEBERG+Cosine, and Oracle GT-MCES-NN (model-free/checkpoint-
# independent, not rerun here) in NOTES_RETRIEVAL_014_2_CORN.md's results
# table.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA+ICEBERG Retrieval: log-loss a40 (raw + CORN-corrected) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_logloss_a40
CHECKPOINT=/sofia/projects/2026_053/simba_project/experiments/training/014_2_logloss_a40_1gpu/checkpoint-epoch=22-step=221000.ckpt
MGF=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test.mgf
CANDIDATES=/sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json
CANDIDATE_TSV_1=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_existing_overlap.tsv
CANDIDATE_TSV_2=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_new.tsv
ICEBERG_PREDS_1=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_test_official/preds.hdf5
ICEBERG_PREDS_2=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_gaetan_test_new/preds.hdf5

mkdir -p "${OUTPUT_DIR}"

cd "${SIMBA_DIR}"

echo "--- raw regression ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${CHECKPOINT}" \
    --head_mode cosine_no_head \
    --mces_bucket_use_mlp \
    --precursor_mass_mode theoretical \
    --min_peaks 6 \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test \
    --batch_size 512 \
    --skip_mces \
    --output_tsv "${OUTPUT_DIR}/retrieval_results_raw.tsv"

echo "--- CORN-corrected ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${CHECKPOINT}" \
    --head_mode cosine_no_head \
    --corn_corrected \
    --mces_bucket_use_mlp \
    --precursor_mass_mode theoretical \
    --min_peaks 6 \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test \
    --batch_size 512 \
    --skip_mces \
    --output_tsv "${OUTPUT_DIR}/retrieval_results_corn.tsv"

echo "===== Retrieval complete: $(date) ====="
