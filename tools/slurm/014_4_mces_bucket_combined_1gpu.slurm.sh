#!/bin/bash
#SBATCH -J simba_train_014_4_bucket_combined
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/014_4_mces_bucket_combined_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/014_4_mces_bucket_combined_1gpu/%x_%j.err

# Experiment 014_4 (replaces the original learnable_weight attempt):
# identical CLI config to experiment 013, combining all three single-
# variable findings from 014_1/2/3's truncated first-epoch data (see
# BASELINE_AND_DASHBOARD.md and conversation log) into one run:
#   - mces_bucket.loss_weight=0.1   (014_1's change)
#   - mces_bucket.use_mlp=true      (014_2's change)
#   - mces_bucket.use_product=true  (014_3's change)
#
# Why: at step 10000 (the only comparable data point, since 014_1-4's
# original attempts died at the epoch boundary from the CSVLogger bug --
# see below), capacity (014_2 bal.acc=0.447, 014_3 bal.acc=0.440) was a far
# bigger lever than the weight change alone (014_1 bal.acc=0.334, vs 013's
# own ~0.32), and the original learnable-weight attempt (014_4 v1,
# bal.acc=0.287) looked worse on both the bucket task and the primary task
# than every other variant at that step. Rather than continue testing the
# isolated-learnable-weight code path further before it's shown any sign of
# working, this run instead asks the more promising question directly:
# does combining the two capacity boosts with the lower fixed weight beat
# any one of them alone? This is no longer a clean single-variable test
# (it bundles three changes at once) -- deliberate, given the goal here is
# "best guess given what we've seen" rather than continued isolation.
#
# The original 014_4 (mces_bucket.learnable_weight=true) code path still
# exists (SimilarityModelMultitask, see mces_bucket_learnable_weight) and
# its crashed first attempt's data is left in
# experiments/training/014_4_mces_bucket_learnable_weight_1gpu/ for the
# record -- not resumed here, but not deleted either.
#
# CSVLogger fix: 014_1/2/3's original attempts (and a second, unpatched
# relaunch of them from an interrupted session) both crashed on the same
# underlying PyTorch Lightning CSVLogger bug (_rewrite_with_new_header
# choking when the on-disk header and the writer's own key list desync --
# seen in two different forms: a stray None key from a malformed re-read,
# and, on the second crash, a genuine missing-keys mismatch, both at the
# same crash site). Patched via
# simba.workflows.training._patch_csv_logger_header_rewrite_resilience
# (called at the top of train()) before this run.
#
# Also unchanged from 013 (which itself = 012 + mces_bucket.enabled=true):
# use_edit_distance=false, cosine_no_head primary head, Gaetan-split-v2
# data, 1x H200, bs=2048, 32-true, lr=0.0001, 24 epochs, resampling on
# (train only), multitasking.learnable=false. Bucket scheme: merged 6-class
# edges [2,4,6,8] (same as 014_1/2/3's reruns), not 013's original 7-class
# [1,2,4,6,8].

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 014_4 (combined): 013 + mces_bucket.loss_weight=0.1 + use_mlp=true + use_product=true ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/014_4_mces_bucket_combined_1gpu

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
  model.tasks.mces_bucket.loss_weight=0.1 \
  model.tasks.mces_bucket.use_mlp=true \
  model.tasks.mces_bucket.use_product=true

echo "===== Training complete: $(date) ====="
