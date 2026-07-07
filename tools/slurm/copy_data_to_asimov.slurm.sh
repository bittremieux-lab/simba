#!/bin/bash
#SBATCH -J copy_data_asimov
#SBATCH -p one_hour
#SBATCH --nodelist=asimov
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -o /home/nkubrakov/simba/logs/copy_data_asimov_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/copy_data_asimov_%j.err

set -uo pipefail

echo "===== Copy data to asimov local disk ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

SRC_BASE=/mnt/data2/nkubrakov
DST_BASE=/mnt/data/nkubrakov

mkdir -p "${DST_BASE}/massspecgym/data/auxiliary"
mkdir -p "${DST_BASE}/massspecgym"
mkdir -p "${DST_BASE}/experiments_3_dataset/training"

echo "--- MGF file ---"
rsync -av --progress \
  "${SRC_BASE}/massspecgym/data/auxiliary/MassSpecGym.mgf" \
  "${DST_BASE}/massspecgym/data/auxiliary/"

echo "--- Preprocessing dir (mces40) ---"
rsync -av --progress \
  "${SRC_BASE}/massspecgym/preprocessing_msg_scaffold_split_mces40" \
  "${DST_BASE}/massspecgym/"

echo "--- Training artifacts (mces40) ---"
rsync -av --progress \
  "${SRC_BASE}/experiments_3_dataset/training/msg_scaffold_split_mces40" \
  "${DST_BASE}/experiments_3_dataset/training/"

echo "===== Copy complete: $(date) ====="
du -sh "${DST_BASE}/massspecgym" "${DST_BASE}/experiments_3_dataset/training/msg_scaffold_split_mces40"
