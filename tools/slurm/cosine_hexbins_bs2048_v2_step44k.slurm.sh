#!/bin/bash
#SBATCH -J cosine_hexbins_bs2048_v2_44k
#SBATCH -p one_day
#SBATCH --nodelist=asimov
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o /home/nkubrakov/simba/logs/cosine_hexbins_bs2048_v2_44k_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/cosine_hexbins_bs2048_v2_44k_%j.err

set -uo pipefail

HEXBIN_DIR=/mnt/data/nkubrakov/experiments_3_dataset/training/msg_exact_mces_1020_no_meta_own_val_weights_bs2048_v2/val_hexbin_step44k
OUTPUT=/home/nkubrakov/simba/results/cosine_hexbins_bs2048_v2_step44k.png

echo "===== Cosine hexbins: bs2048_v2 · step 44k ====="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Date: $(date)"

START=$(date +%s)
cd /home/nkubrakov/simba

uv run python tools/plot_cosine_hexbins.py \
    --hexbin_dir "${HEXBIN_DIR}" \
    --output "${OUTPUT}"

END=$(date +%s)
echo "Wall time: $((END - START))s  Done: $(date)"
