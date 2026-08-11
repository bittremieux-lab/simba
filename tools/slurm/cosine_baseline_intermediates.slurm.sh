#!/bin/bash
#SBATCH -J cosine_baseline_intermediates
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
# -o/-e must be a literal path (target dir must already exist before submission).
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/cosine_baseline_intermediates/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/cosine_baseline_intermediates/%x_%j.err

# Item 8a: precompute binned-spectrum ("cosine baseline", no SIMBA)
# intermediates (tools/cosine_baseline_intermediates.py) for the SAME
# official-test spectra + ICEBERG-predicted candidates used by 008_2's
# retrieval_iceberg run — CPU-only (no model, just binning); requests a GPU
# node purely for fast, uncontended storage I/O.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== Cosine baseline intermediates (item 8a) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/cosine_baseline_intermediates
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
CANDIDATE_TSV=/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_test_official.tsv
ICEBERG_PREDS=/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_test_official/preds.hdf5

mkdir -p "${OUTPUT_DIR}"

cd "${SIMBA_DIR}"

uv run python tools/cosine_baseline_intermediates.py \
    --mgf "${MGF}" \
    --candidate_tsv "${CANDIDATE_TSV}" \
    --iceberg_preds "${ICEBERG_PREDS}" \
    --split test \
    --intermediates_dir "${OUTPUT_DIR}"

echo "===== Done: $(date) ====="
