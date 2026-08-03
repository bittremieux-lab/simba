#!/bin/bash
#SBATCH -J simba_train_008_2_cosine_no_head
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
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/008_2_cosine_no_head_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/008_2_cosine_no_head_1gpu/%x_%j.err

# Experiment 008_2_cosine_no_head: one entry in the head_mode ablation series (008_1..008_5).
# Same official-split, fixed-MCES dataset and sampling/model config as
# experiments 006/007 (MCES==20 excluded, identity pairs, no metadata), except:
#   - model.tasks.cosine_similarity.head_mode=cosine_no_head
#     no projection layers; cosine similarity directly on encoder embeddings (Sentence-BERT / MS2DeepScore style)
#     See simba/core/models/similarity_models.py: HEAD_MODES /
#     compute_from_embeddings.
#   - training.epochs=24 (3x the 8 used in 004/005/006/007), to give the less
#     mature head variants more room to converge.
#   - Single GPU, not the 4x H200 DDP setup used in 004-007: devices=1, no DDP
#     strategy, no srun launcher (that was only needed to make Slurm start
#     the 4 tasks DDP requires). Reverted to the original single-GPU
#     hyperparameters this repo validated before the DDP/bf16 scaling work
#     (see tools/slurm/train_msg_official.slurm.sh's comments): batch_size=2048,
#     limit_train_batches=10000, limit_val_batches=100, lr=0.0001, precision
#     back to the default 32-true (bf16-mixed was introduced together with the
#     4-GPU scaling, not validated standalone on 1 GPU). This reproduces the
#     same per-epoch data volume as before (steps x batch_size), just as many
#     small steps on one GPU instead of fewer large-batch steps split across 4.
#     NOTE: this means ~8x more optimizer steps per epoch than the 4-GPU
#     scripts at 1/4 the per-step throughput (no data parallelism) — expect
#     these runs to take substantially longer in wall-clock time; the
#     24-hour time limit below may not be enough to reach all 24 epochs.

set -uo pipefail

module load uv

# Not derived from ${BASH_SOURCE[0]}: Slurm executes a spooled copy of this
# script on the compute node, so that path doesn't point at the repo.
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 008_2_cosine_no_head: head_mode=cosine_no_head · official split, fixed MCES · no metadata · 1x H200 · bs=2048 · 32-true · lr=0.0001 · 24 epochs ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_official_split_max_lb_hdf5
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/008_2_cosine_no_head_1gpu

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
  model.tasks.cosine_similarity.head_mode=cosine_no_head

echo "===== Training complete: $(date) ====="
