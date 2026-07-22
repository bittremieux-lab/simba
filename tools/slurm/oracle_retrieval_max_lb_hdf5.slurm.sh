#!/bin/bash
#SBATCH -J oracle_retrieval_hdf5fix
#SBATCH -p one_day
#SBATCH --nodelist=login
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -o /home/nkubrakov/simba/logs/oracle_retrieval_hdf5fix_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/oracle_retrieval_hdf5fix_%j.err

# Re-run oracle with HDF5 SMILES canonicalization fix applied.
# Must run on login (needs /mnt/data2 for lb_matrix + HDF5).

set -uo pipefail

echo "===== Oracle Retrieval (HDF5 fix) ====="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Date: $(date)"

START=$(date +%s)
cd /home/nkubrakov/simba

uv run python tools/oracle_retrieval_max_lb_hdf5.py

END=$(date +%s)
echo "Wall time: $((END - START))s  Done: $(date)"
