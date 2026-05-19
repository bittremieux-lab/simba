#!/bin/bash
#SBATCH -J simba_train_th20_asb
#SBATCH -p one_day
#SBATCH --nodelist=asimov2
#SBATCH --gpus=nvidia_h200_nvl_4g.71gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH -o /home/nkubrakov/simba/logs/simba_train_th20_asb_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_train_th20_asb_%j.err

set -uo pipefail

echo "===== SIMBA Training: th20_asb ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
nvidia-smi

PREPRO_DIR=/mnt/data2/nkubrakov/massspecgym/preprocessing_th20_asb
CHECKPOINT_DIR=/mnt/data2/nkubrakov/massspecgym/checkpoints_th20_asb

mkdir -p "$CHECKPOINT_DIR"

cd /home/nkubrakov/simba
source .venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

simba train \
  paths.preprocessing_dir="$PREPRO_DIR" \
  paths.preprocessing_dir_train="$PREPRO_DIR" \
  paths.preprocessing_pickle_file=mapping.pkl \
  paths.checkpoint_dir="$CHECKPOINT_DIR" \
  training.epochs=1000 \
  training.batch_size=3072 \
  training.val_check_interval=1000 \
  training.limit_train_batches=10000 \
  training.limit_val_batches=500 \
  optimizer.lr=0.0001 \
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=10 \
  logging.enable_progress_bar=false \
  logging.log_every_n_steps=10

echo "===== Training complete: $(date) ====="
