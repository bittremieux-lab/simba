#!/bin/bash
#SBATCH -J simba_retrieval_006_step4750
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
# -o/-e must be a literal path (target dir must already exist before submission).
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/006_msg_official_split_max_lb_hdf5_excl_mces20_identity_bf16_4gpu/retrieval_step4750/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/006_msg_official_split_max_lb_hdf5_excl_mces20_identity_bf16_4gpu/retrieval_step4750/%x_%j.err

# Retrieval eval for experiment 006, using checkpoint-epoch=03-step=5000.ckpt —
# the closest saved checkpoint to the epoch=3/step=4750 validation point, which
# had the highest val_mces_spearman/official (0.731) of any epoch in this run
# (the validation itself ran at step=4750, but only every-1000-step checkpoints
# were saved; step=4000 precedes it, step=5000 follows it by 250 steps).

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Retrieval: experiment 006 (official split, fixed MCES), checkpoint epoch=3/step=5000 ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXPERIMENT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/006_msg_official_split_max_lb_hdf5_excl_mces20_identity_bf16_4gpu
OUTPUT_DIR="${EXPERIMENT_DIR}/retrieval_step4750"
CHECKPOINT="${EXPERIMENT_DIR}/checkpoint-epoch=03-step=5000.ckpt"
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
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
