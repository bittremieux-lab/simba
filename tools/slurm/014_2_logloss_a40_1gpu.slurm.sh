#!/bin/bash
#SBATCH -J simba_train_014_2_logloss_a40
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/014_2_logloss_a40_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/014_2_logloss_a40_1gpu/%x_%j.err

# Log-loss sweep (Gaetan's idea, Slack 2026-08-27 -- see
# NOTES_014_2_ANALOG_DISCOVERY.md): identical config to 014_2
# (tools/slurm/014_2_mces_bucket_mlp_1gpu.slurm.sh) in every respect except
# model.tasks.mces.use_log_loss=true. a=40 uses the STRONGER warp Gaetan specifically
# proposed -- pseudocount=1, i.e. training toward log(MCES+1). The
# companion run 014_2_logloss_a5_1gpu.slurm.sh instead keeps a=5 (the
# historical default, pseudocount=8) -- comparing these two plus the 014_2
# baseline (log-loss off entirely) isolates "does log-loss help at all"
# from "does the stronger compression specifically help".
#
# Verified before launching (see simba/core/models/similarity_models.py):
# validation_step computes val_mces_mae and everything ValMetricsCallback
# records from RAW, unwarped model output (logits2) vs the raw target,
# entirely separate from the log-warped variables used only inside step()
# for the loss scalar -- so this stays comparable to every other 014_2
# experiment on the same evaluation metrics, only the training gradient
# signal changes. train and val share the same step() call, so the warp is
# symmetric across both.
#
# Data/config otherwise identical to 014_2: Gaetan-split-v2, cosine_no_head,
# mces_bucket.use_mlp=true, 24 epochs (unchanged -- this run uses the full
# train set, no exclusion filter, so no reason to change the epoch budget).

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 014_2 + log-loss (a=40, pseudocount=1) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/014_2_logloss_a40_1gpu

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
  model.tasks.mces_bucket.use_mlp=true \
  model.tasks.mces.use_log_loss=true \
  model.tasks.mces.log_loss_a=40

echo "===== Training complete: $(date) ====="
