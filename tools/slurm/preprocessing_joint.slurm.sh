#!/bin/bash
#SBATCH --job-name=simba_prepro_joint
#SBATCH --output=logs/prepro_joint_%A_%a.out
#SBATCH --error=logs/prepro_joint_%A_%a.err
#SBATCH -p one_day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=200G
#SBATCH --array=0-1
#SBATCH --nodelist=asimov2

# Joint preprocessing (MSG + NIST20 + Spectraverse) — Murcko scaffold split.
# Precomputed cache: all 3 individual preprocessing dirs → only cross-dataset pairs computed fresh.
#
# Usage:
#   cd /home/nkubrakov/simba
#   mkdir -p logs
#   sbatch tools/slurm/preprocessing_joint.slurm.sh

set -e

SIMBA_DIR=/home/nkubrakov/simba
MGF_PATH=/mnt/data2/nkubrakov/joint/joint_msg_nist20_sv.mgf
OUTPUT_DIR=/mnt/data2/nkubrakov/joint/preprocessing_scaffold_v1
PICKLE_FILE=${OUTPUT_DIR}/mapping.pkl

PRECOMPUTED_MSG=/mnt/data2/nkubrakov/massspecgym/preprocessing_scaffold_v2
PRECOMPUTED_NIST20=/mnt/data2/nkubrakov/nist20/preprocessing
PRECOMPUTED_SV=/mnt/data2/nkubrakov/spectraverse/preprocessing_scaffold_v1

mkdir -p ${SIMBA_DIR}/logs
mkdir -p ${OUTPUT_DIR}

cd ${SIMBA_DIR}

echo "=============================================="
echo "Joint preprocessing (MSG+NIST20+SV) — scaffold_v1, node ${SLURM_ARRAY_TASK_ID}/2"
echo "Host         : $(hostname)"
echo "Array job    : ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "MGF          : ${MGF_PATH}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Cache dirs   : MSG, NIST20, Spectraverse"
echo "Started      : $(date)"
echo "=============================================="

START=$(date +%s)

NUM_WORKERS=${NUM_WORKERS:-50}

uv run simba preprocess \
    paths.spectra_path=${MGF_PATH} \
    paths.preprocessing_dir=${OUTPUT_DIR}/ \
    paths.preprocessing_pickle_file=${PICKLE_FILE} \
    preprocessing.num_workers=${NUM_WORKERS} \
    preprocessing.num_nodes=2 \
    preprocessing.current_node=${SLURM_ARRAY_TASK_ID} \
    preprocessing.max_spectra_train=-1 \
    preprocessing.max_spectra_val=-1 \
    preprocessing.max_spectra_test=-1 \
    preprocessing.force_scaffold_split=True \
    preprocessing.overwrite=False \
    'preprocessing.precomputed_distances=["'${PRECOMPUTED_MSG}'/","'${PRECOMPUTED_NIST20}'/","'${PRECOMPUTED_SV}'/"]'

END=$(date +%s)

echo "=============================================="
echo "Done: $(date)"
echo "Wall time: $((END - START))s"
echo ""
echo "Output .npy files:"
ls -lh ${OUTPUT_DIR}/*.npy 2>/dev/null | head -20 || echo "  (none found)"
echo "=============================================="
