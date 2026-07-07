#!/bin/bash
#SBATCH --job-name=simba_prepro_th20_asb
#SBATCH --output=logs/prepro_th20_asb_%A_%a.out
#SBATCH --error=logs/prepro_th20_asb_%A_%a.err
#SBATCH -p one_day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=256G
#SBATCH --array=0-1
#SBATCH --nodelist=asimov2

# MassSpecGym preprocessing — TH=20, always_stronger_bound=True, single node.
# Fresh output directory, no reuse of previous runs.
#
# Usage:
#   mkdir -p logs
#   sbatch tools/slurm/script_massspecgym_preprocessing_th20_asb.slurm.sh

set -e

SIMBA_DIR=/home/nkubrakov/simba
MGF_PATH=/mnt/data2/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf
HDF5_PATH=/mnt/data2/nkubrakov/massspecgym/data/auxiliary/all_smiles_mces.hdf5
OUTPUT_DIR=/mnt/data2/nkubrakov/massspecgym/preprocessing_th20_asb
PICKLE_FILE=${OUTPUT_DIR}/mapping.pkl

mkdir -p ${SIMBA_DIR}/logs
mkdir -p ${OUTPUT_DIR}

cd ${SIMBA_DIR}

echo "=============================================="
echo "MassSpecGym preprocessing — TH=20, asb=True, node ${SLURM_ARRAY_TASK_ID}/2"
echo "Host       : $(hostname)"
echo "Array job  : ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "Output dir : ${OUTPUT_DIR}"
echo "HDF5 cache : ${HDF5_PATH}"
echo "Started    : $(date)"
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
    preprocessing.hdf5_mces_cache_path=${HDF5_PATH} \
    preprocessing.hdf5_mces_threshold=10.0 \
    preprocessing.overwrite=False

END=$(date +%s)

echo "=============================================="
echo "Done: $(date)"
echo "Wall time: $((END - START))s"
echo ""
echo "Output .npy files:"
ls -lh ${OUTPUT_DIR}/*.npy 2>/dev/null | head -20 || echo "  (none found)"
echo "=============================================="
