#!/bin/bash
#SBATCH -J simba_retrieval_004_fixed
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
# -o/-e must be a literal path (target dir must already exist before submission).
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/004_msg_exact_mces_1020_excl_mces20_identity_bf16_4gpu/retrieval_fixed/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/004_msg_exact_mces_1020_excl_mces20_identity_bf16_4gpu/retrieval_fixed/%x_%j.err

# Retrieval eval for experiment 004 via tools/simba_retrieval.py: SIMBA-embedding
# nearest-neighbor transfer + Tanimoto candidate ranking. Spectra are loaded
# unfiltered (every spectrum in the MGF fold), preprocessed the same way as
# validation (precursor-peak removal, then the intensity floor/top-N cut).

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Retrieval: experiment 004 (excl MCES==20, +identity pairs) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXPERIMENT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/004_msg_exact_mces_1020_excl_mces20_identity_bf16_4gpu
OUTPUT_DIR="${EXPERIMENT_DIR}/retrieval_fixed"
CHECKPOINT="${EXPERIMENT_DIR}/best_model.ckpt"
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
# Formula-matched candidate pool (the mass-matched pool isn't available on this machine).
CANDIDATES=/sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json

mkdir -p "${OUTPUT_DIR}"

cd "${SIMBA_DIR}"

uv run python tools/simba_retrieval.py \
    --checkpoint "${CHECKPOINT}" \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --split test \
    --batch_size 512 \
    --intermediates_dir "${OUTPUT_DIR}" \
    --output_tsv "${OUTPUT_DIR}/retrieval_results.tsv"

echo "===== Retrieval complete: $(date) ====="
