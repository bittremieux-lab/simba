#!/bin/bash
#SBATCH -J simba_retrieval_009_013
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/%x_%j.err

# ICEBERG retrieval (raw only -- none of 009-013 trained a bucket head with
# use_mlp=true, so there's no CORN-corrected variant to compute for them,
# same treatment across all five for a consistent comparison) for the
# 009-013 ablation series, on the same gaetan_test split/candidate set as
# the 014_2 family, so results land in the same dashboard columns.
#
# 009 additionally gets plain (non-CORN) NN-transfer retrieval -- it's the
# closest predecessor to 005 in this series (same head_mode=cosine_no_head,
# just Gaetan-split-v2 data), so it's worth the full 6-column picture
# (ICEBERG + NN) rather than just ICEBERG's 3 like 010-013.
#
# precursor_mass_mode matters here and differs across the series: 009 was
# trained BEFORE sampling.precursor_mass_mode=theoretical was introduced
# (010 onward), so it stayed on the config default "measured" -- passing
# "theoretical" for 009 would silently score it against a precursor mass it
# was never trained to expect. 010-013 all explicitly set
# sampling.precursor_mass_mode=theoretical during training and must be
# scored the same way. Checked each training SLURM script directly, not
# assumed.
#
# None of 009-013 change model.transformer.d_model/n_layers (still the
# 256/5 default), so no --d_model/--n_layers override needed.
#
# Checkpoint choice: latest available step for each (not best_model.ckpt),
# matching the precedent set by 014_2/bigmodel/iceberg-aug/logloss_a40 --
# all of which are the direct comparison set in the dashboard for this
# series.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
MGF=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test.mgf
CANDIDATES=/sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json
CANDIDATE_TSV_1=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_existing_overlap.tsv
CANDIDATE_TSV_2=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_new.tsv
ICEBERG_PREDS_1=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_test_official/preds.hdf5
ICEBERG_PREDS_2=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_gaetan_test_new/preds.hdf5
OUT_ROOT=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test

echo "===== ICEBERG retrieval: 009-013, plus NN-transfer for 009 ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "${SIMBA_DIR}"

CKPT_009=/sofia/projects/2026_053/simba_project/experiments/training/009_msg_gaetan_split_v2_cosine_no_head_1gpu/checkpoint-epoch=15-step=160000.ckpt
CKPT_010=/sofia/projects/2026_053/simba_project/experiments/training/010_msg_gaetan_split_v2_theoretical_precursor_mist_cf_1gpu/checkpoint-epoch=13-step=139000.ckpt
CKPT_011=/sofia/projects/2026_053/simba_project/experiments/training/011_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_1gpu/checkpoint-epoch=15-step=159000.ckpt
CKPT_012=/sofia/projects/2026_053/simba_project/experiments/training/012_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_bucket_weights_1gpu/checkpoint-epoch=17-step=174000.ckpt
CKPT_013=/sofia/projects/2026_053/simba_project/experiments/training/013_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_bucket_weights_mces_bucket_head_1gpu/checkpoint-epoch=22-step=229000.ckpt

mkdir -p "${OUT_ROOT}/simba_iceberg_009" "${OUT_ROOT}/simba_iceberg_010" \
         "${OUT_ROOT}/simba_iceberg_011" "${OUT_ROOT}/simba_iceberg_012" \
         "${OUT_ROOT}/simba_iceberg_013" "${OUT_ROOT}/simba_nn_009"

echo "--- 009: ICEBERG (precursor_mass_mode=measured) ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${CKPT_009}" \
    --head_mode cosine_no_head \
    --precursor_mass_mode measured \
    --min_peaks 6 \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test --batch_size 512 --skip_mces \
    --output_tsv "${OUT_ROOT}/simba_iceberg_009/retrieval_results.tsv"

echo "--- 010: ICEBERG (precursor_mass_mode=theoretical) ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${CKPT_010}" \
    --head_mode cosine_no_head \
    --precursor_mass_mode theoretical \
    --min_peaks 6 \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test --batch_size 512 --skip_mces \
    --output_tsv "${OUT_ROOT}/simba_iceberg_010/retrieval_results.tsv"

echo "--- 011: ICEBERG (precursor_mass_mode=theoretical) ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${CKPT_011}" \
    --head_mode cosine_no_head \
    --precursor_mass_mode theoretical \
    --min_peaks 6 \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test --batch_size 512 --skip_mces \
    --output_tsv "${OUT_ROOT}/simba_iceberg_011/retrieval_results.tsv"

echo "--- 012: ICEBERG (precursor_mass_mode=theoretical) ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${CKPT_012}" \
    --head_mode cosine_no_head \
    --precursor_mass_mode theoretical \
    --min_peaks 6 \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test --batch_size 512 --skip_mces \
    --output_tsv "${OUT_ROOT}/simba_iceberg_012/retrieval_results.tsv"

echo "--- 013: ICEBERG (precursor_mass_mode=theoretical) ---"
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${CKPT_013}" \
    --head_mode cosine_no_head \
    --precursor_mass_mode theoretical \
    --min_peaks 6 \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test --batch_size 512 --skip_mces \
    --output_tsv "${OUT_ROOT}/simba_iceberg_013/retrieval_results.tsv"

echo "--- 009: plain NN-transfer (precursor_mass_mode=measured) ---"
uv run python tools/simba_retrieval.py \
    --checkpoint "${CKPT_009}" \
    --head_mode cosine_no_head \
    --precursor_mass_mode measured \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --split test \
    --batch_size 256 \
    --output_tsv "${OUT_ROOT}/simba_nn_009/retrieval_results.tsv"

echo "===== Retrieval complete: $(date) ====="
