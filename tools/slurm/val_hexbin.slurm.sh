#!/bin/bash
#SBATCH -J simba_val_hexbin
#SBATCH -p one_hour
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH -o /home/nkubrakov/simba/logs/simba_val_hexbin_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_val_hexbin_%j.err

set -uo pipefail

echo "===== SIMBA Val Hexbin ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C /home/nkubrakov/simba rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C /home/nkubrakov/simba rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/mnt/data/nkubrakov/massspecgym/preprocessing_msg_scaffold_split_mces40
EXP_DIR=/mnt/data/nkubrakov/experiments_3_dataset/training/msg_scaffold_split_mces40
CHECKPOINT="${EXP_DIR}/checkpoint-epoch=02-step=29000.ckpt"
OUTPUT_DIR="/mnt/data2/nkubrakov/experiments_3_dataset/training/msg_scaffold_split_mces40/val_hexbin_step29k"

mkdir -p "$OUTPUT_DIR"
mkdir -p /home/nkubrakov/simba/logs

cd /home/nkubrakov/simba

uv run python tools/run_val_hexbin.py \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --prepro_dir "${PREPRO_DIR}" \
  --batch_size 3072 \
  --num_workers 8 \
  model.features.use_adduct=false \
  model.features.use_ce=false \
  model.features.use_ion_mode=false \
  model.multitasking.learnable=false \
  model.tasks.edit_distance.enabled=false \
  model.tasks.edit_distance.n_classes=11

echo "===== Inference done: $(date) ====="

# 6 plots: all pairs
uv run python tools/plot_val_hexbin_balanced.py \
  --val_dir "${OUTPUT_DIR}"

# 6 plots: only pairs with GT MCES <= 20 (comparable to previous run)
uv run python tools/plot_val_hexbin_balanced.py \
  --val_dir "${OUTPUT_DIR}" \
  --mces_max 20

echo "===== Done: $(date) ====="
echo "Outputs in: ${OUTPUT_DIR}"
