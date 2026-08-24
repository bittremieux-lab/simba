#!/bin/bash
#SBATCH -J simba_train_014_3_bucket_product
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/014_3_mces_bucket_product_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/014_3_mces_bucket_product_1gpu/%x_%j.err

# Experiment 014_3: identical CLI config to experiment 013 -- the ONLY
# change is model.tasks.mces_bucket.use_product: false -> true (use_mlp
# stays false, loss_weight stays 1.0, learnable_weight stays false).
#
# Hypothesis: distinct from 014_2's "does depth help" test -- this asks
# whether an explicit multiplicative interaction feature (emb0*emb1,
# concatenated onto |emb0-emb1| before the final linear head) gives the
# bucket head a richer input to work with, independent of adding any
# nonlinear depth. Kept as its own single-variable experiment rather than
# combined with use_mlp=true, so credit for any improvement can be
# attributed to one specific capacity change, not a bundle of two.
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

echo "===== SIMBA Training 014_3: 013 + mces_bucket.use_product=true (was false) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/014_3_mces_bucket_product_1gpu

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
  model.tasks.mces_bucket.use_product=true

echo "===== Training complete: $(date) ====="
