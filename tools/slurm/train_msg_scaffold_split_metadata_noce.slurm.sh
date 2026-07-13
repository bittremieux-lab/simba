#!/bin/bash
#SBATCH -J simba_train_meta_noce
#SBATCH -p one_day
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -o /home/nkubrakov/simba/logs/simba_train_meta_noce_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_train_meta_noce_%j.err

# clip-40 + metadata, CE fix REVERTED (loader reads only params["ce"], so MSG spectra get CE=0).
# Control run to confirm that removing the fix reproduces job 7636 curves.
# 8 epochs (~80k steps) for quick comparison.

set -uo pipefail

echo "===== SIMBA Training: MSG scaffold split — metadata, CE fix REVERTED ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C /home/nkubrakov/simba rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C /home/nkubrakov/simba rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/mnt/data/nkubrakov/massspecgym/preprocessing_msg_scaffold_split_mces40
OUTPUT_DIR=/mnt/data/nkubrakov/experiments_3_dataset/training/msg_scaffold_split_mces40_metadata_noce

mkdir -p "$OUTPUT_DIR"
mkdir -p /home/nkubrakov/simba/logs

cd /home/nkubrakov/simba

export PYTORCH_ALLOC_CONF=expandable_segments:True

uv run simba train \
  paths.preprocessing_dir="${PREPRO_DIR}" \
  paths.preprocessing_dir_train="${PREPRO_DIR}" \
  paths.preprocessing_pickle_file=mapping.pkl \
  paths.checkpoint_dir="${OUTPUT_DIR}" \
  training.epochs=8 \
  training.batch_size=1024 \
  training.val_check_interval=1000 \
  training.limit_train_batches=10000 \
  training.limit_val_batches=500 \
  training.early_stopping_patience=0 \
  optimizer.lr=0.0001 \
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=14 \
  logging.enable_progress_bar=false \
  logging.log_every_n_steps=10 \
  model.features.use_adduct=true \
  model.features.use_ce=true \
  model.features.use_ion_mode=true \
  model.multitasking.learnable=false \
  model.tasks.edit_distance.enabled=false \
  model.tasks.edit_distance.n_classes=11

echo "===== Training complete: $(date) ====="
