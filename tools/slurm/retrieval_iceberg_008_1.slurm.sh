#!/bin/bash
#SBATCH -J simba_retrieval_iceberg_008_1
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/008_1_cosine_baseline_1gpu/retrieval_iceberg/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/008_1_cosine_baseline_1gpu/retrieval_iceberg/%x_%j.err

# SIMBA+ICEBERG retrieval for 008_1 (head_mode=cosine_relu), checkpoint=checkpoint-epoch=01-step=11000.ckpt.
# hit@k only (--skip_mces) — GT-MCES-to-truth stats are still being fixed
# (dedup by unique molecule + per-pair time limit + proper SLURM sizing)
# before being added here; see tools/simba_retrieval_iceberg.py.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA+ICEBERG Retrieval: 008_1 (head_mode=cosine_relu) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXPERIMENT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/008_1_cosine_baseline_1gpu
OUTPUT_DIR="${EXPERIMENT_DIR}/retrieval_iceberg"
CHECKPOINT="${EXPERIMENT_DIR}/checkpoint-epoch=01-step=11000.ckpt"
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
CANDIDATES=/sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json
CANDIDATE_TSV=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_test_official.tsv
ICEBERG_PREDS=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_test_official/preds.hdf5

mkdir -p "${OUTPUT_DIR}"

cd "${SIMBA_DIR}"

uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${CHECKPOINT}" \
    --head_mode cosine_relu \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV}" \
    --iceberg_preds "${ICEBERG_PREDS}" \
    --split test \
    --batch_size 512 \
    --skip_mces \
    --intermediates_dir "${OUTPUT_DIR}" \
    --output_tsv "${OUTPUT_DIR}/retrieval_results.tsv"

echo "===== Retrieval complete: $(date) ====="
