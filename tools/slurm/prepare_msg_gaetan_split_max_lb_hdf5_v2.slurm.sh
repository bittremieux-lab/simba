#!/bin/bash
#SBATCH -J prepare_msg_gaetan_split_max_lb_hdf5_v2
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2_%j.err

# Corrected re-run of prepare_msg_gaetan_split_max_lb_hdf5.py: fixes the HDF5
# SMILES canonicalization bug (query side was canonicalized, the HDF5 lookup
# wasn't, so hdf5_missing was ~94-97% instead of the true near-100%) and
# switches lb_matrix from mmap to a full in-RAM load (the scattered per-pair
# reads this script does turned an equivalent mmap'd lookup elsewhere into a
# 2+ hour stall; this script's original run happened to be fast, but that
# was luck). Output goes to a new _v2 directory -- the original is left
# untouched since experiment 005 was trained on it.
#
# No --mem: this partition doesn't allow it directly (fixed 24-cpus-per-GPU
# ratio, memory follows cpu count). 24 cpus' default allocation (~182 GB)
# comfortably fits the 116 GB matrix plus overhead, same as the earlier
# oracle_retrieval_gt_mces.py fix.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== prepare_msg_gaetan_split_max_lb_hdf5 (v2, HDF5 canonicalization fix) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

cd "${SIMBA_DIR}"

uv run python tools/prepare_msg_gaetan_split_max_lb_hdf5.py

echo "===== Done: $(date) ====="
