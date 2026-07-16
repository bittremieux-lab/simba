#!/bin/bash
#SBATCH -J simba_diag_check_pairs
#SBATCH -p one_hour
#SBATCH --nodelist=asimov
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o /home/nkubrakov/simba/logs/simba_diag_check_pairs_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_diag_check_pairs_%j.err

set -uo pipefail

echo "===== Pair sanity check ====="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Date: $(date)"

PREPRO_DIR=/mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020

cd /home/nkubrakov/simba

uv run python tools/diag_check_pairs.py "${PREPRO_DIR}"

echo "===== Done: $(date) ====="
