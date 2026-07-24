#!/bin/bash
#SBATCH -J simba_train_mces_4gpu_bs16384_bf16
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
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/msg_exact_mces_1020_no_meta_own_val_weights_bs16384_4gpu_bf16/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/msg_exact_mces_1020_no_meta_own_val_weights_bs16384_4gpu_bf16/%x_%j.err

# MCES-only training: no edit distance head, no metadata, own val weights.
# 4x H200 via Lightning DDP. Launched with `srun` so Slurm actually starts the
# 4 tasks --ntasks-per-node=4 reserves — without srun, only 1 task ever runs
# and Lightning's SLURMEnvironment hangs forever waiting for 3 peers that were
# never launched. Each of the 4 srun tasks is its own rank managing exactly
# ONE GPU (hardware.devices=1 below) — Lightning derives world_size=4 from
# Slurm's own env vars, it does not self-spawn sub-processes in this mode.
# This cluster does NOT auto-scope CUDA_VISIBLE_DEVICES per task (all 4 tasks
# see all 4 GPUs identically), and Lightning doesn't correct for that on its
# own — every rank defaulted to physical GPU 0, leaving 1-3 idle. Fixed below
# by explicitly setting CUDA_VISIBLE_DEVICES=$SLURM_LOCALID per task so each
# rank claims exactly one distinct GPU.
# batch_size is per-process under DDP. Kept at the same 2048 that was validated
# on a single GPU (plenty of memory headroom: ~11GB/143GB used at batch=512),
# so the effective global batch is now 8192 (4x the single-GPU baseline) rather
# than matching it exactly — deliberate, to better saturate each GPU instead of
# leaving them mostly idle. limit_train_batches/limit_val_batches (2500/25) are
# per-process too; with batch_size=2048 x 4 ranks they reproduce the exact same
# per-epoch data volume as the single-GPU run's (10000/100 x 2048 x 1 rank).
# NOTE: an earlier version of this script divided batch_size by 4 as well
# (512), which under-scaled the per-epoch volume by an extra 4x on top of this
# — that bug is fixed by restoring batch_size to 2048 here.
# optimizer.lr scaled via sqrt(global-batch-ratio): base 0.0001 at batch=2048,
# global batch is now 16384 -> sqrt(16384/2048)=2.83x -> 0.00028. Adam has no
# warmup/scheduler configured here, so full linear scaling risked early
# instability; sqrt is the standard more conservative rule for Adam-family
# optimizers without a warmup.
# hardware.precision=bf16-mixed: the previous bs=2048/bf32 run only used
# ~29% of each H200's 143GB (~41GB peak) and averaged ~50-57% compute
# utilization (see gpu_util_*.csv from that run) — both compute and memory
# headroom were being left on the table. bf16 roughly halves activation
# memory and raises tensor-core throughput on H200; batch_size is doubled
# (2048 -> 4096/rank) to use the freed-up memory, and limit_train_batches/
# limit_val_batches are halved (2500/25 -> 1250/13) to keep the same
# per-epoch data volume as before rather than silently training on 2x more
# data per epoch.

set -uo pipefail

module load uv

# Not derived from ${BASH_SOURCE[0]}: Slurm executes a spooled copy of this
# script on the compute node, so that path doesn't point at the repo.
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training: exact MCES [10-20] · no metadata · own val weights · 4x H200 DDP · global bs=16384 · bf16-mixed · lr=0.00028 ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_msg_exact_mces_1020
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/msg_exact_mces_1020_no_meta_own_val_weights_bs16384_4gpu_bf16

mkdir -p "$OUTPUT_DIR"

cd "${SIMBA_DIR}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Background GPU utilization/memory logging, one row per GPU every 30s.
GPU_LOG="${OUTPUT_DIR}/gpu_util_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 30 > "${GPU_LOG}" &
GPU_MONITOR_PID=$!
trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null' EXIT

srun bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; exec "$@"' bash \
  uv run simba train \
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
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=14 \
  hardware.precision=bf16-mixed \
  logging.enable_progress_bar=false \
  logging.log_every_n_steps=10 \
  model.features.use_adduct=false \
  model.features.use_ce=false \
  model.features.use_ion_mode=false \
  model.multitasking.learnable=false \
  model.tasks.edit_distance.enabled=false \
  model.tasks.edit_distance.n_classes=11

echo "===== Training complete: $(date) ====="
