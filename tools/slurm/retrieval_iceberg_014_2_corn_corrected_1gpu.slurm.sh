#!/bin/bash
#SBATCH -J simba_retrieval_iceberg_014_2_corn
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_014_2_corn/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_014_2_corn/%x_%j.err

# SIMBA+ICEBERG retrieval for 014_2 (head_mode=cosine_no_head,
# mces_bucket.use_mlp=true), ranked by CORN-corrected MCES instead of plain
# cosine similarity -- see tools/simba_retrieval_iceberg.py's
# rank_candidates_corn_corrected (pairwise bucket-head forward pass per
# query's ICEBERG candidate pool, since the bucket head is a function of the
# pair, not decomposable into cached per-spectrum embeddings the way the
# primary cosine-similarity ranking is).
#
# Reuses the ORIGINAL v1 Gaetan-test ICEBERG candidate files -- confirmed
# byte-identical mapping.pkl between v1 and v2 (same MD5/SHA256), so the
# test fold, candidate pool, and ICEBERG predictions never actually changed;
# no GPU rebuild needed. Query-side precursor mass is switched to
# theoretical (--precursor_mass_mode theoretical) to match how 014_2 itself
# was trained (every checkpoint from experiment 010 onward used
# sampling.precursor_mass_mode=theoretical) -- ICEBERG's own candidate
# spectra were already always theoretical (hardcoded), so only the query
# side needed this fix. --min_peaks 6 additionally reports hit@k restricted
# to test spectra with >= 6 peaks, SIMBA's own canonical minimum-peak-count
# filter (simba/configs/data/default.yaml + all three data-prep scripts).

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA+ICEBERG Retrieval: 014_2 CORN-corrected (head_mode=cosine_no_head) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_014_2_corn
CHECKPOINT=/sofia/projects/2026_053/simba_project/experiments/training/014_2_mces_bucket_mlp_1gpu/checkpoint-epoch=22-step=229000.ckpt
MGF=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test.mgf
CANDIDATES=/sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json
CANDIDATE_TSV_1=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_existing_overlap.tsv
CANDIDATE_TSV_2=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_new.tsv
ICEBERG_PREDS_1=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_test_official/preds.hdf5
ICEBERG_PREDS_2=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_gaetan_test_new/preds.hdf5

mkdir -p "${OUTPUT_DIR}"

cd "${SIMBA_DIR}"

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
    --output_tsv "${OUTPUT_DIR}/retrieval_results.tsv"

echo "===== Retrieval complete: $(date) ====="
