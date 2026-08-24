#!/bin/bash
#SBATCH -J simba_nn_transfer_014_2
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_nn_014_2/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_nn_014_2/%x_%j.err

# SIMBA-NN row (row 2 of NOTES_RETRIEVAL_SPLIT_COMPARISON.md's original table),
# recomputed with 014_2 instead of checkpoint 005 -- two variants, matching
# the ICEBERG rows' treatment:
#   1. Plain embedding NN-transfer (same method as the original 005 run:
#      embed train+test, cosine dot-product NN search, transfer Morgan FP,
#      Tanimoto-rank candidates) -- head_mode=cosine_no_head.
#   2. CORN-corrected NN-transfer: for each test spectrum, find the train
#      spectrum with the lowest CORN-corrected MCES (pairwise bucket-head
#      forward pass) instead of highest cosine similarity -- genuinely
#      pairwise across test x train (~339M pairs after train dedup to
#      24,010 unique molecules), but only the small bucket-head MLP runs
#      pairwise; the transformer encoding is still one pass per spectrum,
#      cached (tools/simba_retrieval.py's nearest_neighbor_transfer_corn_corrected).
#
# Cosine-NN and Oracle-GT-MCES-NN rows are checkpoint-independent and don't
# need recomputation -- the Gaetan-split mapping.pkl is byte-identical
# between v1/v2, so those existing numbers already stand.
#
# --precursor_mass_mode theoretical matches how 014_2 itself was trained
# (every checkpoint from experiment 010 onward).

set -uo pipefail
module load uv
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
cd "${SIMBA_DIR}"

echo "===== SIMBA-NN transfer: 014_2 (plain + CORN-corrected) ====="
echo "Job ID: $SLURM_JOB_ID"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CHECKPOINT=/sofia/projects/2026_053/simba_project/experiments/training/014_2_mces_bucket_mlp_1gpu/checkpoint-epoch=22-step=229000.ckpt
MGF=/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test.mgf
CANDIDATES=/sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json

echo ""
echo "--- Variant 1: plain cosine NN-transfer ---"
uv run python tools/simba_retrieval.py \
    --checkpoint "${CHECKPOINT}" \
    --head_mode cosine_no_head \
    --precursor_mass_mode theoretical \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --split test \
    --batch_size 256 \
    --output_tsv /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_nn_014_2/retrieval_results.tsv

echo ""
echo "--- Variant 2: CORN-corrected NN-transfer ---"
uv run python tools/simba_retrieval.py \
    --checkpoint "${CHECKPOINT}" \
    --head_mode cosine_no_head \
    --corn_corrected \
    --mces_bucket_use_mlp \
    --precursor_mass_mode theoretical \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --split test \
    --batch_size 256 \
    --output_tsv /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_nn_014_2_corn/retrieval_results.tsv

echo "===== Done: $(date) ====="
