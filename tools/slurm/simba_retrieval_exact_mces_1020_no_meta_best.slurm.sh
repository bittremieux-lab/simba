#!/bin/bash
#SBATCH -J simba_ret_exact_mces_no_meta
#SBATCH -p one_day
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /home/nkubrakov/simba/logs/simba_ret_exact_mces_no_meta_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_ret_exact_mces_no_meta_%j.err

# Retrieval benchmark for job 8011: exact MCES [10-20] · no metadata · last checkpoint (step 36k).

set -uo pipefail

SIMBA_DIR=/home/nkubrakov/simba
CHECKPOINT=/mnt/data/nkubrakov/experiments_3_dataset/training/msg_exact_mces_1020_no_meta/checkpoint-epoch=03-step=36000.ckpt
MGF=/mnt/data/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf
CANDIDATES=/mnt/data/nkubrakov/massspecgym/data/molecules/MassSpecGym_retrieval_candidates_mass.json

INTERMEDIATES_DIR=/mnt/data/nkubrakov/experiments_3_dataset/retrieval/exact_mces_1020_no_meta_step36k
OUTPUT_TSV=/home/nkubrakov/simba/results/simba_retrieval_exact_mces_1020_no_meta_step36k.tsv

mkdir -p /home/nkubrakov/simba/results

echo "===== SIMBA Retrieval: exact MCES no metadata · best checkpoint ====="
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
