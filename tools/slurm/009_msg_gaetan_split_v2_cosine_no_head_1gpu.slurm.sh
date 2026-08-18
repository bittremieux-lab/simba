#!/bin/bash
#SBATCH -J simba_train_009_gaetan_v2_cosine_no_head
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
# -o/-e must be a literal path (SBATCH directives are parsed before the script
# runs, so they can't reference $OUTPUT_DIR below) — keep the two in sync, and
# the target dir must already exist before this is submitted (Slurm opens
# these files prior to the script body's own `mkdir -p "$OUTPUT_DIR"`).
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/009_msg_gaetan_split_v2_cosine_no_head_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/009_msg_gaetan_split_v2_cosine_no_head_1gpu/%x_%j.err

# Experiment 009: re-run of experiment 005 (Gaetan's split), adapted three ways:
#   1. Corrected data: preprocessing_gaetan_split_max_lb_hdf5_v2 (the HDF5
#      canonicalization fix -- hdf5_missing went from ~94-97% to 0, see
#      tools/prepare_msg_gaetan_split_max_lb_hdf5.py's module docstring),
#      not the original preprocessing_gaetan_split_max_lb_hdf5 005 used.
#   2. head_mode=cosine_no_head, matching 008_2 (no projection layers, cosine
#      similarity directly on encoder embeddings) -- 005 left head_mode at
#      its default (cosine_relu).
#   3. Single GPU, not the 4x H200 DDP setup 005 used: devices=1, no DDP
#      strategy, no srun launcher. Hyperparameters here are NOT a guess --
#      copied from the validated 1-GPU template used consistently across
#      ALL of 008_1..008_9 (batch_size=2048, lr=0.0001, precision=32-true,
#      strategy=auto, epochs=24, limit_train_batches=10000,
#      limit_val_batches=100) rather than naively shrinking 005's 4-GPU
#      numbers (batch_size=4096/gpu, lr=0.00028, bf16-mixed) by 4x.
#
# Deliberately NOT set here (unlike both 005 and 008_2): sampling.exclude_mces_value.
# 005/008_2 both drop pairs sitting exactly on the MCES=20 classification
# boundary; this run keeps them (left at the config default, null).
#
# Kept from 005 (unchanged): sampling.add_identity_pairs=true, no ED task
# (model.tasks.edit_distance.enabled=false), no learnable multitask weighting
# (model.multitasking.learnable=false), no adduct/CE/ion-mode metadata.
#
# New relative to 005/008_2, requested before this launch:
#   4. sampling.use_resampling=false -- disables the inverse-MCES-bin-frequency
#      weighted sampler for both train and val (was silently always-on before;
#      this flag existed in config but was never wired up). Train falls back to
#      plain shuffling; val falls back to a single deterministic unweighted pass.
#   5. training.limit_val_batches is NOT set (was 100 in the 008_* template) --
#      every validation check now scores the full ~3.9M-pair val set instead of
#      a ~5% weighted-resampled slice, needed for the per-MCES-bin MAE stats and
#      per-pair CSV dump below to be meaningful.
#   6. Validation now logs per-MCES-bin MAE (self-pairs get their own bin, not
#      folded into the lowest numeric bin) instead of Spearman, saves a full
#      per-pair CSV every epoch, and saves a GT-binned predicted-MCES boxplot
#      every epoch -- see ValMetricsCallback in simba/core/training/callbacks.py.
#      No ED confusion matrix is saved (the ED head isn't scored here anyway).

set -uo pipefail

module load uv

# Not derived from ${BASH_SOURCE[0]}: Slurm executes a spooled copy of this
# script on the compute node, so that path doesn't point at the repo.
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 009: Gaetan split v2 · head_mode=cosine_no_head · 1x H200 · bs=2048 · 32-true · lr=0.0001 · 24 epochs · no MCES==20 exclusion · no resampling · full val ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/009_msg_gaetan_split_v2_cosine_no_head_1gpu

mkdir -p "$OUTPUT_DIR"

cd "${SIMBA_DIR}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Background GPU utilization/memory logging, one row every 30s.
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
  sampling.use_resampling=false \
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
  model.tasks.cosine_similarity.head_mode=cosine_no_head

echo "===== Training complete: $(date) ====="
