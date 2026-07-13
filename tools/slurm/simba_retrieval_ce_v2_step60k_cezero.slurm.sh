#!/bin/bash
#SBATCH -J simba_ret_ce_v2_60k_ce0
#SBATCH -p one_day
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /home/nkubrakov/simba/logs/simba_ret_ce_v2_step60k_cezero_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_ret_ce_v2_step60k_cezero_%j.err

# Diagnostic: retrieval with CE forced to 0 for all spectra.
# Tests whether CE-induced embedding collapse is why retrieval is poor.

set -uo pipefail

SIMBA_DIR=/home/nkubrakov/simba
CHECKPOINT=/mnt/data/nkubrakov/experiments_3_dataset/training/msg_scaffold_split_mces40_metadata_ce_v2/checkpoint-epoch=05-step=60000.ckpt
MGF=/mnt/data/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf
CANDIDATES=/mnt/data/nkubrakov/massspecgym/data/molecules/MassSpecGym_retrieval_candidates_mass.json

INTERMEDIATES_DIR=/mnt/data/nkubrakov/experiments_3_dataset/retrieval/msg_ce_v2_step60k_cezero
OUTPUT_TSV=/home/nkubrakov/simba/results/simba_retrieval_ce_v2_step60k_cezero.tsv

mkdir -p /home/nkubrakov/simba/results

echo "===== SIMBA Retrieval: CE v2 step 60k · CE=0 diagnostic ====="
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
    --output_tsv "${OUTPUT_TSV}" \
    --force_ce_zero

END=$(date +%s)
echo "Wall time: $((END - START))s  Done: $(date)"
