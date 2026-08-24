#!/bin/bash
#SBATCH -J simba_train_013_mces_bucket_head
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/013_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_bucket_weights_mces_bucket_head_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/013_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_bucket_weights_mces_bucket_head_1gpu/%x_%j.err

# Experiment 013: identical CLI config to experiment 012 -- the only change
# is adding the new optional second target, model.tasks.mces_bucket
# (a CORN-style ordinal classification head trained in parallel on MCES,
# independent of cosine_similarity.head_mode). Bins: 0 (self-pairs), then
# (0,1], (1,2], (2,4], (4,6], (6,8], (8,inf) -- 7 classes total, using the
# config's defaults (bin_edges=[1,2,4,6,8], use_mlp=false, use_product=false).
#
# model.multitasking.learnable stays false (matching 012) -- deliberately
# NOT switched to learnable=true for this run, to isolate this one variable:
# switching learnable weighting would also change how the *primary* loss2 is
# scaled (from weight=1 to an initial ~200x under log_sigma2's init), which
# would confound this being a clean "012 + one new task" comparison. So the
# new task's contribution is combined via the fixed weight
# model.tasks.mces_bucket.loss_weight (default 1.0, added on top of loss2
# with no scale-matching) -- whether that default weight over- or under-
# balances the two losses is itself one of the things this run is meant to
# surface, not something to hand-tune preemptively.
#
# use_edit_distance=false (unchanged from 012) also exercises the new
# "skip the ED head's forward pass and loss entirely when disabled" fix
# (previously it was computed and discarded every batch even when unused).
#
# New validation-time artifacts from this task (see BASELINE_AND_DASHBOARD.md):
#   - val_mces_bucket_balanced_acc/{val_name} in metrics.csv
#   - mces_bucket_confusion_{val_name}_step{N:06d}.png (counts + row% + col%)
#   - pred_mces_bucket_step{N:06d} column in val_pairs_{val_name}_consolidated.parquet
#   - loss_mces_bucket (train+val) now logged, alongside the previously-silent
#     loss_ed/loss_mces (a pre-existing gap fixed alongside this feature)
#
# Unchanged from 012: Gaetan-split-v2 data, 1x H200, bs=2048, 32-true,
# lr=0.0001, 24 epochs, no MCES==20 exclusion, add_identity_pairs=true,
# no adduct/CE/ion-mode metadata, sampling.precursor_mass_mode=theoretical,
# sampling.precursor_noise_mode=mist_cf, sampling.use_resampling=true
# (training only), cosine_no_head primary head.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 013: 012 + optional MCES-bucket second target (CORN-style ordinal head, 7 classes) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/013_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_bucket_weights_mces_bucket_head_1gpu

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
  model.tasks.mces_bucket.enabled=true

echo "===== Training complete: $(date) ====="
