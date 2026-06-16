#!/bin/bash
#SBATCH --job-name=simba_prepro_spectraverse
#SBATCH --output=logs/prepro_spectraverse_%A_%a.out
#SBATCH --error=logs/prepro_spectraverse_%A_%a.err
#SBATCH -p one_day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=200G
#SBATCH --array=0-1
#SBATCH --nodelist=asimov2

# Spectraverse preprocessing — TH=20, always_stronger_bound=True, Murcko scaffold splits.
# Precomputed cache: MSG scaffold_v2 (covers 16.1% of SV pairs, 40.2% of SV molecules).
# No HDF5 cache.
#
# Usage:
#   cd /home/nkubrakov/simba
#   mkdir -p logs
#   sbatch tools/slurm/preprocessing_spectraverse.slurm.sh

set -euo pipefail

SIMBA_DIR=/home/nkubrakov/simba
MGF_PATH=/mnt/data2/nkubrakov/spectraverse/spectraverse-1.0.1.mgf
PRECOMPUTED_DIR=/mnt/data2/nkubrakov/massspecgym/preprocessing_scaffold_v2
OUTPUT_DIR=/mnt/data2/nkubrakov/spectraverse/preprocessing_scaffold_v1
PICKLE_FILE=${OUTPUT_DIR}/mapping.pkl

mkdir -p ${SIMBA_DIR}/logs
mkdir -p ${OUTPUT_DIR}

cd ${SIMBA_DIR} || exit 1

echo "=============================================="
echo "Spectraverse preprocessing — scaffold splits v1, node ${SLURM_ARRAY_TASK_ID}/2"
echo "Host         : $(hostname)"
echo "Array job    : ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "MGF          : ${MGF_PATH}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Precomputed  : ${PRECOMPUTED_DIR}"
echo "force_scaffold_split=True"
echo "Started      : $(date)"
echo "=============================================="

START=$(date +%s)

uv run simba preprocess \
    paths.spectra_path=${MGF_PATH} \
    paths.preprocessing_dir=${OUTPUT_DIR}/ \
    paths.preprocessing_pickle_file=${PICKLE_FILE} \
    preprocessing.num_workers=50 \
    preprocessing.num_nodes=2 \
    preprocessing.current_node=${SLURM_ARRAY_TASK_ID} \
    preprocessing.max_spectra_train=-1 \
    preprocessing.max_spectra_val=-1 \
    preprocessing.max_spectra_test=-1 \
    preprocessing.force_scaffold_split=True \
    preprocessing.overwrite=False \
    'preprocessing.precomputed_distances=["'${PRECOMPUTED_DIR}'/"]'

END=$(date +%s)

echo "=============================================="
echo "Done: $(date)"
echo "Wall time: $((END - START))s"
echo ""
echo "Output .npy files:"
ls -lh ${OUTPUT_DIR}/*.npy 2>/dev/null | head -20 || echo "  (none found)"
echo "=============================================="
