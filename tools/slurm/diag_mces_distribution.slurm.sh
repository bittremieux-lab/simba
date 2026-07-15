#!/bin/bash
#SBATCH -J simba_diag_mces
#SBATCH -p one_hour
#SBATCH --nodelist=asimov
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o /home/nkubrakov/simba/logs/simba_diag_mces_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_diag_mces_%j.err

set -uo pipefail

echo "===== MCES distribution diagnostic ====="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Date: $(date)"

PREPRO_DIR=/mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020

cd /home/nkubrakov/simba

uv run python tools/diag_mces_distribution.py "${PREPRO_DIR}"

echo "===== Done: $(date) ====="
