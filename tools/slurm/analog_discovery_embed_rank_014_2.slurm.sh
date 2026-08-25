#!/bin/bash
#SBATCH -J analog_discovery_embed_rank_014_2
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/data/analog_discovery/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/data/analog_discovery/%x_%j.err

# 014_2 analog discovery, stage 2: embed CASMI queries + one reference
# library with 014_2 (head_mode=cosine_no_head, mces_bucket.use_mlp=true)
# and score every query x library molecule pair under three schemes
# (SIMBA raw regression, SIMBA CORN-corrected, plain binned-peak cosine) --
# see tools/analog_discovery_embed_rank.py and NOTES_014_2_ANALOG_DISCOVERY.md.
#
# --precursor_mass_mode theoretical matches how 014_2 itself was trained
# (sampling.precursor_mass_mode=theoretical, every checkpoint from
# experiment 010 onward).
#
# Run once per SEARCH argument ("A" or "B") -- submit both:
#   sbatch --export=SEARCH=A tools/slurm/analog_discovery_embed_rank_014_2.slurm.sh
#   sbatch --export=SEARCH=B tools/slurm/analog_discovery_embed_rank_014_2.slurm.sh
#
# Requires tools/prepare_analog_discovery_data.py to have already been run
# (produces data/analog_discovery/search_A_nist_msg.mgf / search_B_gnps.mgf).

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
DATA_DIR=/sofia/projects/2026_053/simba_project/data/analog_discovery
CHECKPOINT=/sofia/projects/2026_053/simba_project/experiments/training/014_2_mces_bucket_mlp_1gpu/checkpoint-epoch=22-step=229000.ckpt

if [ -z "${SEARCH:-}" ]; then
    echo "SEARCH not set -- submit with: sbatch --export=SEARCH=A|B $0" >&2
    exit 1
fi

case "${SEARCH}" in
    A) MGF="${DATA_DIR}/search_A_nist_msg.mgf"; OUTPUT_DIR="${DATA_DIR}/search_A_scores" ;;
    B) MGF="${DATA_DIR}/search_B_gnps.mgf"; OUTPUT_DIR="${DATA_DIR}/search_B_scores" ;;
    *) echo "SEARCH must be A or B, got '${SEARCH}'" >&2; exit 1 ;;
esac

echo "===== Analog discovery embed+rank: search ${SEARCH} ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "${OUTPUT_DIR}"

cd "${SIMBA_DIR}"

uv run python tools/analog_discovery_embed_rank.py \
    --checkpoint "${CHECKPOINT}" \
    --mgf "${MGF}" \
    --output_dir "${OUTPUT_DIR}" \
    --head_mode cosine_no_head \
    --mces_bucket_use_mlp \
    --precursor_mass_mode theoretical \
    --batch_size 512

echo "===== Embed+rank complete (search ${SEARCH}): $(date) ====="
