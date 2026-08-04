#!/bin/bash
#SBATCH -J simba_retrieval_iceberg_008_2
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
# -o/-e must be a literal path (target dir must already exist before submission).
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/008_2_cosine_no_head_1gpu/retrieval_iceberg/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/008_2_cosine_no_head_1gpu/retrieval_iceberg/%x_%j.err

# SIMBA+ICEBERG retrieval: score real official-test spectra directly against
# ICEBERG-predicted candidate spectra (tools/simba_retrieval_iceberg.py) — no
# train spectra involved, unlike the NN-transfer retrieval in
# tools/simba_retrieval.py. Uses 008_2 (cosine_no_head), the best-performing
# head_mode from the 008 ablation, at its own best-official-Spearman
# checkpoint (epoch=2/step=24000). Candidate predictions come from
# ICEBERG/results/candidates_test_official/preds.hdf5 (600,455 candidates,
# msg_all weights, single fixed CE=35.0 — see ICEBERG/build_candidate_tsv.py).

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA+ICEBERG Retrieval: 008_2 (cosine_no_head), best-official-spearman checkpoint ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXPERIMENT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/008_2_cosine_no_head_1gpu
OUTPUT_DIR="${EXPERIMENT_DIR}/retrieval_iceberg"
CHECKPOINT="${EXPERIMENT_DIR}/checkpoint-epoch=02-step=24000.ckpt"
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
CANDIDATES=/sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json
CANDIDATE_TSV=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_test_official.tsv
ICEBERG_PREDS=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_test_official/preds.hdf5

mkdir -p "${OUTPUT_DIR}"

cd "${SIMBA_DIR}"

uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint "${CHECKPOINT}" \
    --head_mode cosine_no_head \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --candidate_tsv "${CANDIDATE_TSV}" \
    --iceberg_preds "${ICEBERG_PREDS}" \
    --split test \
    --batch_size 512 \
    --intermediates_dir "${OUTPUT_DIR}" \
    --output_tsv "${OUTPUT_DIR}/retrieval_results.tsv"

echo "===== Retrieval complete: $(date) ====="
