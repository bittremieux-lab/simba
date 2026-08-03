#!/bin/bash
#SBATCH -J simba_retrieval_008_1_cosine_baseline_1gpu
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
# -o/-e must be a literal path (target dir must already exist before submission).
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/008_1_cosine_baseline_1gpu/retrieval_best_official_spearman/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/008_1_cosine_baseline_1gpu/retrieval_best_official_spearman/%x_%j.err

# Retrieval eval for 008_1_cosine_baseline_1gpu (head_mode=cosine_relu), using the checkpoint at
# this run's own best val_mces_spearman/official point (found from
# metrics.csv, not the scaffold-loss-selected best_model.ckpt): checkpoint-epoch=01-step=11000.ckpt.
# tools/simba_retrieval.py's embedding extraction must be told the head_mode
# explicitly — head_mode is not stored in the checkpoint (no
# save_hyperparameters() call), so it defaults to cosine_relu otherwise,
# which would silently apply the wrong projection for every head_mode here
# except 008_1.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Retrieval: 008_1_cosine_baseline_1gpu (head_mode=cosine_relu), checkpoint=checkpoint-epoch=01-step=11000.ckpt ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXPERIMENT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/008_1_cosine_baseline_1gpu
OUTPUT_DIR="${EXPERIMENT_DIR}/retrieval_best_official_spearman"
CHECKPOINT="${EXPERIMENT_DIR}/checkpoint-epoch=01-step=11000.ckpt"
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
    --head_mode cosine_relu \
    --intermediates_dir "${OUTPUT_DIR}" \
    --output_tsv "${OUTPUT_DIR}/retrieval_results.tsv"

echo "===== Retrieval complete: $(date) ====="
