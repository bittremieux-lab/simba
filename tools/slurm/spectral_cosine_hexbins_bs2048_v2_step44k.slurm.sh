#!/bin/bash
#SBATCH -J spec_cos_hexbins_bs2048_v2_44k
#SBATCH -p one_hour
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH -o /home/nkubrakov/simba/logs/spec_cos_hexbins_bs2048_v2_44k_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/spec_cos_hexbins_bs2048_v2_44k_%j.err

# Re-run val hexbin inference to regenerate CSVs with the cosine_spectral column,
# then build 9-panel spectral-cosine hexbin figure.

set -uo pipefail

echo "===== Spectral cosine hexbins: bs2048_v2 @ step 44k ====="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Date: $(date)"
nvidia-smi -L

PREPRO_DIR=/mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020
EXP_DIR=/mnt/data/nkubrakov/experiments_3_dataset/training/msg_exact_mces_1020_no_meta_own_val_weights_bs2048_v2
CHECKPOINT="${EXP_DIR}/checkpoint-epoch=04-step=44000.ckpt"
HEXBIN_DIR="${EXP_DIR}/val_hexbin_step44k"
OUTPUT=/home/nkubrakov/simba/results/cosine_hexbins_bs2048_v2_step44k.png

mkdir -p "$HEXBIN_DIR"
mkdir -p /home/nkubrakov/simba/logs /home/nkubrakov/simba/results

cd /home/nkubrakov/simba

START=$(date +%s)

# 1. Re-run inference — regenerates val_predictions_*.csv with cosine_spectral column
uv run python tools/run_val_hexbin.py \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${HEXBIN_DIR}" \
  --prepro_dir "${PREPRO_DIR}" \
  --batch_size 3072 \
  --num_workers 8 \
  model.features.use_adduct=false \
  model.features.use_ce=false \
  model.features.use_ion_mode=false \
  model.multitasking.learnable=false \
  model.tasks.edit_distance.enabled=false \
  model.tasks.edit_distance.n_classes=11

# 2. Build the 9-panel spectral cosine figure
uv run python tools/plot_cosine_hexbins.py \
  --hexbin_dir "${HEXBIN_DIR}" \
  --output "${OUTPUT}"

END=$(date +%s)
echo "Wall time: $((END - START))s  Done: $(date)"
