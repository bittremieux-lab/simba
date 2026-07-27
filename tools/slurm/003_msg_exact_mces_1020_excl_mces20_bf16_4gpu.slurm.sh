#!/bin/bash
#SBATCH -J simba_train_003_exclmces20
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
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/003_msg_exact_mces_1020_excl_mces20_bf16_4gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/003_msg_exact_mces_1020_excl_mces20_bf16_4gpu/%x_%j.err

# Experiment 003: same as the bf16/bs16384 4x H200 DDP run, but with pairs
# whose raw MCES distance is exactly 20 dropped from BOTH train and val
# (sampling.exclude_mces_value=20 below) — MCES==20 is the classification
# task's threshold (model.tasks.mces.threshold), so pairs sitting exactly on
# that boundary were previously ambiguous/noisy training signal either side
# of the decision edge.
#
# See tools/slurm/train_msg_official.slurm.sh for the full rationale behind
# the DDP/srun launch pattern, bf16 precision, and batch size — unchanged
# here. hardware.devices=4 + hardware.strategy=ddp_find_unused_parameters_true
# + no CUDA_VISIBLE_DEVICES scoping is likewise required (see that script's
# comments for the three ways this breaks otherwise): devices=1/strategy=auto
# silently writes one checkpoint per rank (-v1/-v2/-v3); devices=1/strategy=ddp
# combined with per-task CUDA_VISIBLE_DEVICES scoping crashes outright once
# Lightning's SLURM validation and per-rank device indexing kick in; and
# plain strategy=ddp (without find_unused_parameters) crashes because this
# MCES-only config never backprops through the edit-distance head.

set -uo pipefail

module load uv

# Not derived from ${BASH_SOURCE[0]}: Slurm executes a spooled copy of this
# script on the compute node, so that path doesn't point at the repo.
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 003: exact MCES [10-20], excluding MCES==20 · no metadata · own val weights · 4x H200 DDP · global bs=16384 · bf16-mixed · lr=0.00028 ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_msg_exact_mces_1020
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/003_msg_exact_mces_1020_excl_mces20_bf16_4gpu

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
