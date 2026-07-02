#!/bin/bash
#SBATCH -J simba_preprocess_scaffold
#SBATCH -p one_hour
#SBATCH --nodelist=asimov2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o /home/nkubrakov/simba/logs/simba_preprocess_scaffold_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_preprocess_scaffold_%j.err

set -uo pipefail

echo "===== SIMBA Preprocessing: MSG scaffold split ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C /home/nkubrakov/simba rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C /home/nkubrakov/simba rev-parse --short HEAD)"

cd /home/nkubrakov/simba

uv run python tools/prepare_msg_scaffold_splits.py

echo "===== Preprocessing complete: $(date) ====="
