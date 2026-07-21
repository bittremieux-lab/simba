#!/bin/bash
#SBATCH -J simba_ret_44k_fix_all
#SBATCH -p one_day
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /mnt/data/nkubrakov/experiments_3_dataset/retrieval/bs2048_v2_step44k_fixed_all/slurm_%j.out
#SBATCH -e /mnt/data/nkubrakov/experiments_3_dataset/retrieval/bs2048_v2_step44k_fixed_all/slurm_%j.err

# Three fixes applied:
# 1. n_layers=5 (checkpoint has 5 layers; was incorrectly set to 8 → 3 random layers)
# 2. Peak selection: top-100 by intensity (matching filter_intensity in training)
# 3. Normalization: sqrt+L2 (matching Augmentation.normalize_intensities in __getitem__)

set -uo pipefail

SIMBA_DIR=/home/nkubrakov/simba
CHECKPOINT=/mnt/data/nkubrakov/experiments_3_dataset/training/msg_exact_mces_1020_no_meta_own_val_weights_bs2048_v2/checkpoint-epoch=04-step=44000.ckpt
MGF=/mnt/data/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf
CANDIDATES=/mnt/data/nkubrakov/massspecgym/data/molecules/MassSpecGym_retrieval_candidates_mass.json

INTERMEDIATES_DIR=/mnt/data/nkubrakov/experiments_3_dataset/retrieval/bs2048_v2_step44k_fixed_all
OUTPUT_TSV=/mnt/data/nkubrakov/experiments_3_dataset/retrieval/bs2048_v2_step44k_fixed_all/retrieval_results.tsv

mkdir -p "${INTERMEDIATES_DIR}"

echo "===== SIMBA Retrieval: bs2048_v2 · step 44k · all fixes (n_layers=5, top-N intensity, sqrt+L2) ====="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

START=$(date +%s)
cd ${SIMBA_DIR}

uv run python tools/simba_retrieval.py \
    --checkpoint "${CHECKPOINT}" \
    --mgf "${MGF}" \
    --candidates "${CANDIDATES}" \
    --split test \
    --batch_size 512 \
    --intermediates_dir "${INTERMEDIATES_DIR}" \
    --output_tsv "${OUTPUT_TSV}"

END=$(date +%s)
echo "Wall time: $((END - START))s  Done: $(date)"
