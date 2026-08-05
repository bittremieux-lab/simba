#!/bin/bash
#SBATCH -J mces_exact_retrieval_candidates
#SBATCH -p one_day
#SBATCH --nodelist=asimov2
#SBATCH --array=0-59
# To run a single benchmark block first: sbatch --array=0 mces_exact_retrieval_candidates.slurm.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err

# Exact MCES (threshold=20, always_stronger_bound=True) for official-test
# retrieval candidate pairs (3c/3e) — ~585k (test molecule, candidate
# molecule) pairs, one array task = one block of ~9.7k pairs (60 blocks).
#
# Run tools/prepare_gt_mces_retrieval.py prepare on sofia FIRST, then copy
# its OUTPUT_DIR here before submitting this (see that script's docstring).
#
# Restart safety: blocks with a matching .done file are skipped automatically.
# To resubmit failed blocks only, pass --array=<comma-separated ids> to sbatch.
#
# Monitor progress:
#   uv run python tools/prepare_gt_mces_retrieval.py --output_dir "$OUTPUT_DIR" status
#   watch -n 60 'ls '"$OUTPUT_DIR"'/blocks/*.done | wc -l'

set -uo pipefail

SIMBA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR=/mnt/data2/nkubrakov/mces_exact_retrieval_candidates

echo "===== MCES exact retrieval-candidates block ${SLURM_ARRAY_TASK_ID} ====="
echo "Job  : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node : ${SLURM_NODELIST}  CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Date : $(date)"

cd "${SIMBA_DIR}"

uv run python tools/prepare_gt_mces_retrieval.py \
    --output_dir "${OUTPUT_DIR}" \
    compute_block \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --n_jobs "${SLURM_CPUS_PER_TASK}"

echo "Done : $(date)"
