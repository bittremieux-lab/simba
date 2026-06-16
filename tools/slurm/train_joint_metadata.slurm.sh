#!/bin/bash
#SBATCH -J simba_train_joint_metadata
#SBATCH -p seven_days
#SBATCH --nodelist=asimov2
#SBATCH --gpus=nvidia_h200_nvl_4g.71gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -o /home/nkubrakov/simba-integration/logs/simba_train_joint_metadata_%j.out
#SBATCH -e /home/nkubrakov/simba-integration/logs/simba_train_joint_metadata_%j.err

set -euo pipefail

echo "===== SIMBA Training: Joint dataset + metadata (adduct, CE, ion_mode) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
nvidia-smi

PREPRO_DIR=/mnt/data2/nkubrakov/joint/preprocessing_scaffold_v1
CHECKPOINT_DIR=/mnt/data2/nkubrakov/experiments_3_dataset/training/metadata_adduct_ce_ionmode_attn

mkdir -p "$CHECKPOINT_DIR"
mkdir -p /home/nkubrakov/simba-integration/logs

cd /home/nkubrakov/simba-integration || exit 1

export PYTORCH_ALLOC_CONF=expandable_segments:True

uv run simba train \
  paths.preprocessing_dir="${PREPRO_DIR}" \
  paths.preprocessing_dir_train="${PREPRO_DIR}" \
  paths.preprocessing_pickle_file=mapping.pkl \
  paths.checkpoint_dir="${CHECKPOINT_DIR}" \
  training.epochs=1000 \
  training.batch_size=3072 \
  training.val_check_interval=1000 \
  training.limit_train_batches=10000 \
  training.limit_val_batches=500 \
  training.early_stopping_patience=10 \
  optimizer.lr=0.0001 \
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=14 \
  logging.enable_progress_bar=false \
  logging.log_every_n_steps=10 \
  model.features.use_adduct=true \
  model.features.use_ce=true \
  model.features.use_ion_mode=true

echo "===== Training complete: $(date) ====="
