#!/bin/bash
#SBATCH -J analog_discovery_embed_rank_ckpt
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/data/analog_discovery/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/data/analog_discovery/%x_%j.err

# Generic embed+rank re-run for an arbitrary checkpoint (both searches A and
# B), used for the CASMI-distance-exclusion sweep's grid comparison (see
# NOTES_014_2_ANALOG_DISCOVERY.md) -- LABEL becomes the output dir suffix:
# data/analog_discovery/search_{A,B}_scores_${LABEL}/.
#
# Usage: sbatch --export=CHECKPOINT=/path/to/ckpt,LABEL=excl4 \
#          tools/slurm/analog_discovery_embed_rank_checkpoint.slurm.sh

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
DATA_DIR=/sofia/projects/2026_053/simba_project/data/analog_discovery

if [ -z "${CHECKPOINT:-}" ] || [ -z "${LABEL:-}" ]; then
    echo "CHECKPOINT and LABEL must both be set -- sbatch --export=CHECKPOINT=...,LABEL=... $0" >&2
    exit 1
fi

echo "===== Analog discovery embed+rank: checkpoint ${LABEL} ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Checkpoint: ${CHECKPOINT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "${SIMBA_DIR}"

for SEARCH in A B; do
    case "${SEARCH}" in
        A) MGF="${DATA_DIR}/search_A_nist_msg.mgf" ;;
        B) MGF="${DATA_DIR}/search_B_gnps.mgf" ;;
    esac
    OUTPUT_DIR="${DATA_DIR}/search_${SEARCH}_scores_${LABEL}"
    mkdir -p "${OUTPUT_DIR}"
    echo "--- search ${SEARCH} -> ${OUTPUT_DIR} ---"
    uv run python tools/analog_discovery_embed_rank.py \
        --checkpoint "${CHECKPOINT}" \
        --mgf "${MGF}" \
        --output_dir "${OUTPUT_DIR}" \
        --head_mode cosine_no_head \
        --mces_bucket_use_mlp \
        --precursor_mass_mode theoretical \
        --batch_size 512
done

echo "===== Embed+rank complete (${LABEL}): $(date) ====="
