#!/bin/bash
#SBATCH -J simba_train_014_1_bucket_lower_weight
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/014_1_mces_bucket_lower_weight_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/014_1_mces_bucket_lower_weight_1gpu/%x_%j.err

# Experiment 014_1: identical CLI config to experiment 013 -- the ONLY
# change is model.tasks.mces_bucket.loss_weight: 1.0 -> 0.1 (fixed weight,
# multitasking.learnable and mces_bucket.learnable_weight both stay false).
#
# Hypothesis: at the current weight (1.0, i.e. loss = loss2 + loss3
# unscaled), the auxiliary CORN-bucket task is measurably hurting the
# primary task's convergence -- confirmed directly by comparing 013 to 012
# (identical config, no bucket head) at matched steps: 013's loss_mces /
# val_mces_mae are consistently worse than 012's throughout training (e.g.
# step~35k: 012 loss=0.0259/mae=4.91 vs 013 loss_mces=0.0324/mae=5.66).
# Cutting the bucket task's weight to 0.1 tests whether reducing its
# influence recovers 012-like primary-task numbers. This is a single,
# surgical change -- it does NOT touch how loss1/loss2 combine (that stays
# on the exact same fixed-weight path as 013), only the bucket term's own
# scale.
#
# Also unchanged from 013 (which itself = 012 + mces_bucket.enabled=true):
# use_edit_distance=false, cosine_no_head primary head, Gaetan-split-v2
# data, 1x H200, bs=2048, 32-true, lr=0.0001, 24 epochs, resampling on
# (train only). Bucket scheme note: the config default bin_edges changed
# since 013 launched (empty (1,2] bin merged into (0,2], now [2,4,6,8] / 6
# classes instead of 013's [1,2,4,6,8] / 7 classes) -- not part of this
# experiment's own variable, just inherited from the current code.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 014_1: 013 + mces_bucket.loss_weight=0.1 (was 1.0) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/014_1_mces_bucket_lower_weight_1gpu

mkdir -p "$OUTPUT_DIR"

cd "${SIMBA_DIR}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

GPU_LOG="${OUTPUT_DIR}/gpu_util_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 30 > "${GPU_LOG}" &
GPU_MONITOR_PID=$!
trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null' EXIT

uv run simba train \
  paths.preprocessing_dir="${PREPRO_DIR}" \
  paths.preprocessing_dir_train="${PREPRO_DIR}" \
  paths.preprocessing_pickle_file=mapping.pkl \
  paths.checkpoint_dir="${OUTPUT_DIR}" \
  paths.mgf_path="${MGF}" \
  training.epochs=24 \
  training.batch_size=2048 \
  training.val_check_interval=1000 \
  training.limit_train_batches=10000 \
  training.early_stopping_patience=0 \
  optimizer.lr=0.0001 \
  sampling.add_identity_pairs=true \
  sampling.use_resampling=true \
  sampling.precursor_mass_mode=theoretical \
  sampling.precursor_noise_mode=mist_cf \
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=14 \
  hardware.precision=32-true \
  hardware.strategy=auto \
  logging.enable_progress_bar=false \
  logging.log_every_n_steps=10 \
  model.features.use_adduct=false \
  model.features.use_ce=false \
  model.features.use_ion_mode=false \
  model.multitasking.learnable=false \
  model.tasks.edit_distance.enabled=false \
  model.tasks.edit_distance.n_classes=11 \
  model.tasks.cosine_similarity.head_mode=cosine_no_head \
  model.tasks.mces_bucket.enabled=true \
  model.tasks.mces_bucket.loss_weight=0.1

echo "===== Training complete: $(date) ====="
