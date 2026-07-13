#!/bin/bash
#SBATCH -J mces_exact_1020
#SBATCH -p one_day
#SBATCH --nodelist=asimov2
#SBATCH --array=0-199
# To run a single benchmark block first: sbatch --array=0 mces_exact_1020.slurm.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -o /home/nkubrakov/simba/logs/mces_exact_1020_%A_%a.out
#SBATCH -e /home/nkubrakov/simba/logs/mces_exact_1020_%A_%a.err

# Exact MCES (threshold=20, always_stronger_bound=True) for training pairs
# with lb in [10, 20].  One array task = one block of ~313k pairs.
#
# Restart safety: blocks with a matching .done file are skipped automatically.
# To resubmit failed blocks only, pass --array=<comma-separated ids> to sbatch.
#
# Monitor progress:
#   python tools/compute_mces_exact_1020.py status
#   watch -n 60 'ls /mnt/data2/nkubrakov/mces_exact_10_20/blocks/*.done | wc -l'
#
# Data: all input + output on /mnt/data2 (asimov2 local disk). No /mnt/data access.
# Resources: asimov2 has 128 CPUs → up to 5 concurrent 16-CPU tasks when idle.

set -uo pipefail

SIMBA_DIR=/home/nkubrakov/simba
OUTPUT_DIR=/mnt/data2/nkubrakov/mces_exact_1020

mkdir -p "${SIMBA_DIR}/logs"

echo "===== MCES exact [10,20] block ${SLURM_ARRAY_TASK_ID} ====="
echo "Job  : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node : ${SLURM_NODELIST}  CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Date : $(date)"

cd "${SIMBA_DIR}"

uv run python tools/compute_mces_exact_1020.py \
    --output_dir "${OUTPUT_DIR}" \
    compute_block \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --n_jobs "${SLURM_CPUS_PER_TASK}"

echo "Done : $(date)"
