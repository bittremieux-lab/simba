#!/bin/bash
#SBATCH -J simba_diag_mces_train
#SBATCH -p one_hour
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -o /home/nkubrakov/simba/logs/simba_diag_mces_train_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_diag_mces_train_%j.err

set -uo pipefail

echo "===== MCES distribution diagnostic via training init ====="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Date: $(date)"

PREPRO_DIR=/mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020
OUTPUT_DIR=/mnt/data/nkubrakov/experiments_3_dataset/training/diag_mces_dist

mkdir -p "$OUTPUT_DIR"

cd /home/nkubrakov/simba

uv run simba train \
  paths.preprocessing_dir="${PREPRO_DIR}" \
  paths.preprocessing_dir_train="${PREPRO_DIR}" \
  paths.preprocessing_pickle_file=mapping.pkl \
  paths.checkpoint_dir="${OUTPUT_DIR}" \
  training.epochs=1 \
  training.batch_size=1024 \
  training.val_check_interval=500 \
  training.limit_train_batches=500 \
  training.limit_val_batches=100 \
  training.early_stopping_patience=0 \
  optimizer.lr=0.0001 \
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=14 \
  logging.enable_progress_bar=false \
  logging.log_every_n_steps=1 \
  model.features.use_adduct=false \
  model.features.use_ce=false \
  model.features.use_ion_mode=false \
  model.multitasking.learnable=false \
  model.tasks.edit_distance.enabled=false \
  model.tasks.edit_distance.n_classes=11

echo "===== Done: $(date) ====="
