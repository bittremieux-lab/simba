#!/bin/bash
#SBATCH -J simba_ret_clip40_meta
#SBATCH -p one_day
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /home/nkubrakov/simba/logs/simba_ret_clip40_meta_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_ret_clip40_meta_%j.err

# SIMBA retrieval benchmark: clip-40 + metadata · step 22k (job 7636)
# Note: retrieval script does not pass adduct/CE/ion_mode to the encoder;
# those features will default to zero. Results may be slightly underestimated.

set -uo pipefail

SIMBA_DIR=/home/nkubrakov/simba
CHECKPOINT=/mnt/data/nkubrakov/experiments_3_dataset/training/msg_scaffold_split_mces40_metadata/checkpoint-epoch=02-step=22000.ckpt
MGF=/mnt/data/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf
CANDIDATES=/mnt/data/nkubrakov/massspecgym/data/molecules/MassSpecGym_retrieval_candidates_mass.json

INTERMEDIATES_DIR=/mnt/data/nkubrakov/experiments_3_dataset/retrieval/msg_clip40_metadata_step22k
OUTPUT_TSV=/home/nkubrakov/simba/results/simba_retrieval_clip40_metadata_step22k.tsv

mkdir -p /home/nkubrakov/simba/results

echo "===== SIMBA Retrieval: clip-40 + metadata · step 22k ====="
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
