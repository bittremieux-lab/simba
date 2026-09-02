#!/bin/bash
#SBATCH -J simba_retrieval_iceberg_bigmodel_aug
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/%x_%j.err

# ICEBERG retrieval (raw + CORN-corrected) for the bigger-model and
# ICEBERG-augmentation checkpoints (see NOTES_014_2_BIGMODEL.md,
# NOTES_014_2_ICEBERG_AUGMENTATION.md), to compare against the existing
# 014_2 baseline row in NOTES_RETRIEVAL_014_2_CORN.md. Both checkpoints
# actually hit the 24h SLURM wall-clock TIMEOUT rather than completing all
# 24 epochs naturally -- bigmodel stopped at epoch=13/step=131000 (014_2
# itself reached step=229000, so this is meaningfully less trained, not an
# apples-to-apples maturity comparison); the ICEBERG-augmentation run
# reached epoch=22/step=229000, essentially the same maturity as 014_2's
# own reference checkpoint.
#
# bigmodel needs --d_model 384 --n_layers 8 explicitly (see
# NOTES_014_2_BIGMODEL.md -- the checkpoint doesn't record its own
# architecture size, so this MUST be passed correctly or strict=False
# silently drops every shape-mismatched weight instead of erroring).
# iceberg-aug uses the same small architecture as 014_2, no override
# needed. Same Gaetan-test ICEBERG candidate files as the existing 014_2
# retrieval run (mgf, candidates json, candidate_tsv, iceberg_preds).

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
MGF=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test.mgf
CANDIDATES=/sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json
CANDIDATE_TSV_1=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_existing_overlap.tsv
CANDIDATE_TSV_2=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_new.tsv
ICEBERG_PREDS_1=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_test_official/preds.hdf5
ICEBERG_PREDS_2=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_gaetan_test_new/preds.hdf5

echo "===== ICEBERG retrieval: bigmodel + ICEBERG-augmentation (raw + CORN) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "${SIMBA_DIR}"

BIGMODEL_CKPT=/sofia/projects/2026_053/simba_project/experiments/training/014_2_bigmodel_d384_l8_1gpu/checkpoint-epoch=13-step=131000.ckpt
BIGMODEL_OUT=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_bigmodel
mkdir -p "${BIGMODEL_OUT}"

echo "--- bigmodel: raw regression ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${BIGMODEL_CKPT}" \
    --head_mode cosine_no_head \
    --mces_bucket_use_mlp \
    --precursor_mass_mode theoretical \
    --min_peaks 6 \
    --d_model 384 --n_layers 8 \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test --batch_size 512 --skip_mces \
    --output_tsv "${BIGMODEL_OUT}/retrieval_results_raw.tsv"

echo "--- bigmodel: CORN-corrected ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${BIGMODEL_CKPT}" \
    --head_mode cosine_no_head \
    --corn_corrected \
    --mces_bucket_use_mlp \
    --precursor_mass_mode theoretical \
    --min_peaks 6 \
    --d_model 384 --n_layers 8 \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test --batch_size 512 --skip_mces \
    --output_tsv "${BIGMODEL_OUT}/retrieval_results_corn.tsv"

AUG_CKPT=/sofia/projects/2026_053/simba_project/experiments/training/014_2_iceberg_aug_p50_1gpu/checkpoint-epoch=22-step=229000.ckpt
AUG_OUT=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_augmentation
mkdir -p "${AUG_OUT}"

echo "--- iceberg-aug: raw regression ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${AUG_CKPT}" \
    --head_mode cosine_no_head \
    --mces_bucket_use_mlp \
    --precursor_mass_mode theoretical \
    --min_peaks 6 \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test --batch_size 512 --skip_mces \
    --output_tsv "${AUG_OUT}/retrieval_results_raw.tsv"

echo "--- iceberg-aug: CORN-corrected ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${AUG_CKPT}" \
    --head_mode cosine_no_head \
    --corn_corrected \
    --mces_bucket_use_mlp \
    --precursor_mass_mode theoretical \
    --min_peaks 6 \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test --batch_size 512 --skip_mces \
    --output_tsv "${AUG_OUT}/retrieval_results_corn.tsv"

echo "===== Retrieval complete: $(date) ====="
