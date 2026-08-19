#!/bin/bash
#SBATCH -J simba_train_011_resampling_self_bucket
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/011_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/011_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_1gpu/%x_%j.err

# Experiment 011: identical to experiment 010 (Gaetan-split-v2, 1x H200,
# head_mode=cosine_no_head, theoretical precursor mass + MIST-CF/BUDDY noise),
# except training batches are now drawn via the inverse-MCES-bin-frequency
# weighted sampler instead of plain shuffling:
#
#   sampling.use_resampling=true -- re-enables CustomWeightedRandomSampler for
#   TRAINING ONLY. Validation (scaffold and, if present, official) always
#   scores the full val set once, unweighted, sequentially, regardless of
#   this flag -- val_sampler/val_official_sampler are now unconditionally
#   None in simba/workflows/training.py's prepare_data, so turning resampling
#   back on can no longer silently turn a validation check back into a
#   partial, non-deterministic, weighted-with-replacement slice (which is
#   what happened, unnoticed, every time use_resampling was true before this
#   change) -- it only ever affects which training pairs get seen how often.
#
# The only new thing relative to how resampling worked in 005/008_* (the
# last runs that had it on): the inverse-frequency MCES-bin scheme used to
# lump raw MCES==0 pairs (self-pairs from sampling.add_identity_pairs, and
# any other exactly-identical-structure pair) into the same [0,2.5) bucket
# as near-but-not-identical pairs, so self-pairs never got a weight of their
# own -- just whatever that combined bucket's inverse frequency worked out
# to. There's now a dedicated bucket for exactly MCES=0, splitting the old
# 10-bin scheme into 11 (see simba/workflows/training.py's use_mces_sampling
# branch). Self-pairs are far rarer in absolute count than (0,2.5) pairs, so
# this bucket's own inverse-frequency weight is now meaningfully higher than
# it was while diluted together with them.
#
# Unchanged from 010: Gaetan-split-v2 data, 1x H200, bs=2048, 32-true,
# lr=0.0001, 24 epochs, no MCES==20 exclusion, add_identity_pairs=true, no ED
# task, no learnable multitask weighting, no adduct/CE/ion-mode metadata,
# sampling.precursor_mass_mode=theoretical, sampling.precursor_noise_mode=mist_cf.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 011: Gaetan split v2 · theoretical precursor mass + MIST-CF/BUDDY noise · MCES-weighted resampling (train only) + dedicated self-pair bucket · cosine_no_head · 1x H200 · bs=2048 · 32-true · lr=0.0001 · 24 epochs ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/011_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_1gpu

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
  model.tasks.cosine_similarity.head_mode=cosine_no_head

echo "===== Training complete: $(date) ====="
