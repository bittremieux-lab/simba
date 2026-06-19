#!/bin/bash
#SBATCH --job-name=simba_prepro_nist20
#SBATCH --output=logs/prepro_nist20_%j.out
#SBATCH --error=logs/prepro_nist20_%j.err
#SBATCH -p one_day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=150G
#SBATCH --nodelist=asimov

# NIST20 preprocessing — TH=20, always_stronger_bound=True, Murcko scaffold splits.
# No precomputed cache — all pairs computed fresh by ILP workers.
# Single node (asimov), no array — 56 CPUs, num_nodes=1.
#
# Data is local to asimov (/mnt/data): fast I/O.
# Precomputed cache is on asimov2 NFS (/mnt/data2): read once at startup, OK.
#
# Usage:
#   cd /home/nkubrakov/simba
#   mkdir -p logs
#   sbatch tools/slurm/preprocessing_nist20.slurm.sh

set -euo pipefail

SIMBA_DIR=/home/nkubrakov/simba
MGF_PATH=/mnt/data/nkubrakov/nist20/nist20.mgf
OUTPUT_DIR=/mnt/data/nkubrakov/nist20/preprocessing
PICKLE_FILE=${OUTPUT_DIR}/mapping.pkl

mkdir -p ${SIMBA_DIR}/logs
mkdir -p ${OUTPUT_DIR}

cd ${SIMBA_DIR} || exit 1

echo "=============================================="
echo "NIST20 preprocessing — scaffold splits, single node"
echo "Host         : $(hostname)"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "CPUs         : ${SLURM_CPUS_PER_TASK}"
echo "MGF          : ${MGF_PATH}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "No precomputed cache — full ILP computation"
echo "force_scaffold_split=True"
echo "Started      : $(date)"
echo "=============================================="

START=$(date +%s)

uv run simba preprocess \
    paths.spectra_path=${MGF_PATH} \
    paths.preprocessing_dir=${OUTPUT_DIR}/ \
    paths.preprocessing_pickle_file=${PICKLE_FILE} \
    preprocessing.num_workers=48 \
    preprocessing.num_nodes=1 \
    preprocessing.current_node=0 \
    preprocessing.max_spectra_train=-1 \
    preprocessing.max_spectra_val=-1 \
    preprocessing.max_spectra_test=-1 \
    preprocessing.force_scaffold_split=True \
    preprocessing.overwrite=False

END=$(date +%s)

echo "=============================================="
echo "Done: $(date)"
echo "Wall time: $((END - START))s"
echo ""
echo "Output .npy files:"
ls -lh ${OUTPUT_DIR}/*.npy 2>/dev/null | head -20 || echo "  (none found)"
echo "=============================================="
