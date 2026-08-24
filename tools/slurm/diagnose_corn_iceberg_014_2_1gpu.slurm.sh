#!/bin/bash
#SBATCH -J simba_diagnose_corn_014_2
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_014_2_corn/diagnose_%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_014_2_corn/diagnose_%x_%j.err

# Diagnostic for the CORN-corrected ICEBERG+SIMBA retrieval result: isolates
# whether the gap to ICEBERG+Cosine comes from the bucket-head correction
# specifically, or is already present in the plain embedding-cosine signal
# (the same code path used since checkpoint 005) -- computed from the SAME
# embeddings in one pass. Also prints per-pair arithmetic for a handful of
# real queries for manual verification.

set -uo pipefail
module load uv
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
cd "${SIMBA_DIR}"

echo "===== Diagnose CORN-corrected vs plain-cosine (014_2) ====="
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

uv run python tools/diagnose_corn_iceberg.py \
    --checkpoint "${CHECKPOINT}" \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV_1}" "${CANDIDATE_TSV_2}" \
    --iceberg_preds "${ICEBERG_PREDS_1}" "${ICEBERG_PREDS_2}" \
    --split test \
    --batch_size 512 \
    --n_diagnose 5

echo "===== Diagnostic complete: $(date) ====="
