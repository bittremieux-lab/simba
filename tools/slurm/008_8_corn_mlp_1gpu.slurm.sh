#!/bin/bash
#SBATCH -J simba_train_008_8_corn_mlp
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
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/008_8_corn_mlp_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/008_8_corn_mlp_1gpu/%x_%j.err

# Experiment 008_8_corn_mlp: CORN ordinal-classification head (Shi, Cao & Raschka,
# 2021/2023) on binned raw MCES, replacing the cosine/distance regression
# heads from the 008_1..5 series. small MLP before the CORN logits, |emb0-emb1| only.
# corn_bin_edges=[2,4,6,8,10,16,22,28,34,40]: step 2 below MCES=10, then
# uniform step 6 up to mces.max_value=40 (11 bins total). Same official-split
# fixed-MCES dataset, sampling, and single-GPU hyperparameters as 008_1..5
# (batch_size=2048,
# limit_train_batches=10000, limit_val_batches=100, lr=0.0001, 32-true
# precision, 24 epochs). See simba/core/models/similarity_models.py:
# HEAD_MODES / compute_from_embeddings / _corn_loss / _corn_decode_similarity.

set -uo pipefail

module load uv

# Not derived from ${BASH_SOURCE[0]}: Slurm executes a spooled copy of this
# script on the compute node, so that path doesn't point at the repo.
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 008_8_corn_mlp: head_mode=corn (mlp=true, product=false) · official split, fixed MCES · no metadata · 1x H200 · bs=2048 · 32-true · lr=0.0001 · 24 epochs ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_official_split_max_lb_hdf5
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/008_8_corn_mlp_1gpu

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
  training.limit_val_batches=100 \
  training.early_stopping_patience=0 \
  optimizer.lr=0.0001 \
  sampling.exclude_mces_value=20 \
  sampling.add_identity_pairs=true \
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
  model.tasks.cosine_similarity.head_mode=corn \
  model.tasks.cosine_similarity.corn_bin_edges=[2,4,6,8,10,16,22,28,34,40] \
  model.tasks.cosine_similarity.corn_use_mlp=true \
  model.tasks.cosine_similarity.corn_use_product=false

echo "===== Training complete: $(date) ====="
