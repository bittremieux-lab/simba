#!/bin/bash
#SBATCH -J simba_retrieval
#SBATCH -p one_day
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /home/nkubrakov/simba/logs/simba_retrieval_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_retrieval_%j.err

# SIMBA retrieval benchmark: embedding NN transfer + Tanimoto ranking.
# Model: max(Gaetan lb, HDF5) · MCES only · step 70k
#
# Candidate JSON must already be on /mnt/data before submitting:
#   mkdir -p /mnt/data/nkubrakov/massspecgym/data/molecules
#   cp /mnt/data2/.../MassSpecGym_retrieval_candidates_mass.json /mnt/data/.../
#
# Usage:
#   mkdir -p logs results
#   sbatch tools/slurm/simba_retrieval.slurm.sh

set -uo pipefail

SIMBA_DIR=/home/nkubrakov/simba
CHECKPOINT=/mnt/data/nkubrakov/experiments_3_dataset/training/msg_max_lb_hdf5_mces40/checkpoint-epoch=06-step=70000.ckpt
MGF=/mnt/data/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf
CANDIDATES=/mnt/data/nkubrakov/massspecgym/data/molecules/MassSpecGym_retrieval_candidates_mass.json

INTERMEDIATES_DIR=/mnt/data/nkubrakov/experiments_3_dataset/retrieval/msg_max_lb_hdf5_step70k
OUTPUT_TSV=/home/nkubrakov/simba/results/simba_retrieval_msg_max_lb_hdf5_step70k.tsv

mkdir -p /home/nkubrakov/simba/results

echo "===== SIMBA Retrieval Benchmark ====="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Date:       $(date)"
echo "Checkpoint: $CHECKPOINT"
echo "Branch:     $(git -C ${SIMBA_DIR} rev-parse --abbrev-ref HEAD)"
echo "Commit:     $(git -C ${SIMBA_DIR} rev-parse --short HEAD)"
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
echo ""
echo "Wall time: $((END - START))s"
echo "Done: $(date)"
echo "Results TSV: ${OUTPUT_TSV}"
echo "Embeddings:  ${INTERMEDIATES_DIR}/"
