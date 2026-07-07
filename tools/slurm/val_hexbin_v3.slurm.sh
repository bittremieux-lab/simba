#!/bin/bash
#SBATCH -J simba_val_hexbin_v3
#SBATCH -p one_day
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH -o /home/nkubrakov/simba/logs/simba_val_hexbin_v3_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_val_hexbin_v3_%j.err

set -uo pipefail

echo "===== SIMBA Val Hexbin v3 ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C /home/nkubrakov/simba rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C /home/nkubrakov/simba rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/mnt/data2/nkubrakov/massspecgym/preprocessing_msg_scaffold_split
EXP_DIR=/mnt/data2/nkubrakov/experiments_3_dataset/training/msg_scaffold_split_v3
CHECKPOINT="${EXP_DIR}/checkpoint-epoch=06-step=61000.ckpt"
OUTPUT_DIR="${EXP_DIR}/val_hexbin"

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
  model.tasks.edit_distance.enabled=false

echo "===== Done: $(date) ====="
echo "Outputs in: ${OUTPUT_DIR}"
