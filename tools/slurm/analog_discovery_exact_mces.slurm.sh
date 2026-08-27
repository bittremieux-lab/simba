#!/bin/bash
#SBATCH -J analog_discovery_exact_mces
# ADJUST FOR TARGET CLUSTER — this is meant to run on "another CPU server"
# (per the user's own instruction), so partition/nodelist are placeholders,
# not copied from this repo's asimov2-specific template:
#SBATCH -p one_day
#SBATCH --array=0-99%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err

# 014_2 analog discovery, stage 3: exhaustive exact-MCES(threshold=20) for one
# search's full query x library molecule-pair space
# (tools/analog_discovery_exact_mces.py — see that script's docstring and
# NOTES_014_2_ANALOG_DISCOVERY.md for the full pipeline).
#
# Same restart-safe block/.done pattern as
# tools/slurm/mces_exact_retrieval_candidates.slurm.sh, but this script has
# NO asimov2/metabo_depthcharge dependency — myopic_mces + PULP_CBC_CMD only,
# both plain pip packages, confirmed to work in this repo's own .venv. So
# unlike that template, this one does NOT need --nodelist -- it should run on
# any CPU node with this repo's .venv (or an equivalent env: rdkit, numpy,
# tqdm, myopic_mces) available.
#
# Run tools/analog_discovery_exact_mces.py prepare FIRST (on whichever
# machine has the search_*.mgf file — fast, seconds), then copy OUTPUT_DIR
# (meta.json, smiles.txt, pairs.npy — blocks/ doesn't exist yet) to wherever
# this job runs if different, before submitting.
#
# Usage:
#   sbatch --export=OUTPUT_DIR=/path/to/search_A_exact_mces tools/slurm/analog_discovery_exact_mces.slurm.sh
#   sbatch --export=OUTPUT_DIR=/path/to/search_B_exact_mces tools/slurm/analog_discovery_exact_mces.slurm.sh
#
# --array bounds MUST match the --n_blocks value passed to `prepare` (default
# 100 -> 0-99). To run a single block first as a sanity check:
#   sbatch --array=0 --export=OUTPUT_DIR=... tools/slurm/analog_discovery_exact_mces.slurm.sh
#
# Restart safety: blocks with a matching .done file are skipped automatically
# — resubmitting the same array after a partial run only redoes missing blocks.
#
# Monitor progress:
#   uv run python tools/analog_discovery_exact_mces.py --output_dir "$OUTPUT_DIR" status
#   watch -n 60 'ls '"$OUTPUT_DIR"'/blocks/*.done | wc -l'
#
# Combine when done:
#   uv run python tools/analog_discovery_exact_mces.py --output_dir "$OUTPUT_DIR" combine

set -euo pipefail

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

if [ -z "${OUTPUT_DIR:-}" ]; then
    echo "OUTPUT_DIR not set -- submit with: sbatch --export=OUTPUT_DIR=/path/to/... $0" >&2
    exit 1
fi

echo "===== Analog discovery exact-MCES block ${SLURM_ARRAY_TASK_ID} ====="
echo "Job       : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node      : ${SLURM_NODELIST}  CPUs: ${SLURM_CPUS_PER_TASK}"
echo "OutputDir : ${OUTPUT_DIR}"
echo "Date      : $(date)"

cd "${SIMBA_DIR}"

"${SIMBA_DIR}/.venv/bin/python" tools/analog_discovery_exact_mces.py \
    --output_dir "${OUTPUT_DIR}" \
    compute_block \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --n_jobs "${SLURM_CPUS_PER_TASK}"

echo "Done : $(date)"
