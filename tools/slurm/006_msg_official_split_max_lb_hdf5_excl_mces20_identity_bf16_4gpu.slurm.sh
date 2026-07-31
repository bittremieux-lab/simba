#!/bin/bash
#SBATCH -J simba_train_006_official_split
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
# -o/-e must be a literal path (SBATCH directives are parsed before the script
# runs, so they can't reference $OUTPUT_DIR below) — keep the two in sync, and
# the target dir must already exist before this is submitted (Slurm opens
# these files prior to the script body's own `mkdir -p "$OUTPUT_DIR"`).
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/006_msg_official_split_max_lb_hdf5_excl_mces20_identity_bf16_4gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/006_msg_official_split_max_lb_hdf5_excl_mces20_identity_bf16_4gpu/%x_%j.err

# Experiment 006: same hyperparameters as experiment 004 (MCES==20 excluded,
# identity pairs added, bf16/DDP setup), but trained on
# data/massspecgym/preprocessing_official_split_max_lb_hdf5 —
# tools/prepare_msg_official_split_max_lb_hdf5.py: official MassSpecGym
# train/val_official/test fold labels (official train re-split by Murcko
# scaffold, seed=42, VAL_FRAC=0.10, into train/val — same split methodology
# and molecule counts as the original preprocessing_msg_exact_mces_1020,
# confirmed by matching pair-file byte sizes exactly), MCES = max(lb_matrix,
# HDF5) using the corrected HDF5 index formula, no separate exact-recompute
# pass for the [10,20] band (unlike msg_exact_mces_1020, which additionally
# replaced that band with real ILP computation).
#
# This dataset has train/val(scaffold)/val_official/test, same structure as
# experiments 001-004 — no adaptation needed there, same CLI overrides as 004.
#
# See tools/slurm/train_msg_official.slurm.sh for the full rationale behind
# the DDP/srun launch pattern, bf16 precision, and batch size, and
# tools/slurm/003_msg_exact_mces_1020_excl_mces20_bf16_4gpu.slurm.sh for the
# MCES==20 exclusion rationale — both unchanged here.

set -uo pipefail

module load uv

# Not derived from ${BASH_SOURCE[0]}: Slurm executes a spooled copy of this
# script on the compute node, so that path doesn't point at the repo.
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 006: official split, max(lb_matrix, HDF5) fixed indexing, no 10-20 recompute, excluding MCES==20, +identity pairs · no metadata · 4x H200 DDP · global bs=16384 · bf16-mixed · lr=0.00028 ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_official_split_max_lb_hdf5
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/006_msg_official_split_max_lb_hdf5_excl_mces20_identity_bf16_4gpu

mkdir -p "$OUTPUT_DIR"

cd "${SIMBA_DIR}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Background GPU utilization/memory logging, one row per GPU every 30s.
GPU_LOG="${OUTPUT_DIR}/gpu_util_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 30 > "${GPU_LOG}" &
GPU_MONITOR_PID=$!
trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null' EXIT

srun uv run simba train \
  paths.preprocessing_dir="${PREPRO_DIR}" \
  paths.preprocessing_dir_train="${PREPRO_DIR}" \
  paths.preprocessing_pickle_file=mapping.pkl \
  paths.checkpoint_dir="${OUTPUT_DIR}" \
  paths.mgf_path="${MGF}" \
  training.epochs=8 \
  training.batch_size=4096 \
  training.val_check_interval=1000 \
  training.limit_train_batches=1250 \
  training.limit_val_batches=13 \
  training.early_stopping_patience=0 \
  optimizer.lr=0.00028 \
  sampling.exclude_mces_value=20 \
  sampling.add_identity_pairs=true \
  hardware.accelerator=gpu \
  hardware.devices=4 \
  hardware.num_workers=14 \
  hardware.precision=bf16-mixed \
  hardware.strategy=ddp_find_unused_parameters_true \
  logging.enable_progress_bar=false \
  logging.log_every_n_steps=10 \
  model.features.use_adduct=false \
  model.features.use_ce=false \
  model.features.use_ion_mode=false \
  model.multitasking.learnable=false \
  model.tasks.edit_distance.enabled=false \
  model.tasks.edit_distance.n_classes=11

echo "===== Training complete: $(date) ====="
