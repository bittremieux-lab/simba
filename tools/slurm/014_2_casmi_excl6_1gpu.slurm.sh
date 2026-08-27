#!/bin/bash
#SBATCH -J simba_train_014_2_casmi_excl6
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/014_2_casmi_excl6_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/014_2_casmi_excl6_1gpu/%x_%j.err

# CASMI-distance-exclusion sweep (threshold=6) (see NOTES_014_2_ANALOG_DISCOVERY.md,
# "Follow-up: does SIMBA's edge come from CASMI being close to its training
# data?" -- Wout's proposed test, Slack 2026-08-27): identical config to
# 014_2 (tools/slurm/014_2_mces_bucket_mlp_1gpu.slurm.sh) in every respect
# EXCEPT:
#   - sampling.train_exclude_smiles_file drops every Gaetan-split TRAIN
#     molecule whose min GT-MCES distance to any CASMI query is < 6 (see
#     simba/workflows/training.py::load_dataset -- new opt-in filter, same
#     mol_idx_remap mechanism already used there for missing-spectra drops,
#     val/test untouched). 3507 / 24,010 train molecules excluded, leaving
#     20503 -- see the job log for the exact "kept X / Y pairs" count from
#     _apply_remap in prepare_data().
#   - training.epochs=240 (10x 014_2's 24) -- fewer training pairs means
#     fewer steps/epoch at the same limit_train_batches, so more epochs are
#     needed for a comparable total step budget. No automatic early
#     stopping (early_stopping_patience=0, unchanged from 014_2) -- meant to
#     be monitored and killed manually once validation loss plateaus, same
#     24h SLURM wall-clock ceiling as 014_2 (not expected to run all 240
#     epochs to completion).
# Preprocessing data (PREPRO_DIR) is 014_2's own Gaetan-split-v2 directory,
# completely unmodified -- the exclusion happens at training-data-load time
# via the new sampling.train_exclude_smiles_file option, not by rebuilding
# any preprocessing artifact.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 014_2 CASMI-excl6 (train molecules < MCES 6 from any CASMI query dropped) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/014_2_casmi_excl6_1gpu
EXCLUDE_FILE=/sofia/projects/2026_053/simba_project/data/analog_discovery/train_exclude_smiles_threshold6.txt

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
  training.epochs=240 \
  training.batch_size=2048 \
  training.val_check_interval=1000 \
  training.limit_train_batches=10000 \
  training.early_stopping_patience=0 \
  optimizer.lr=0.0001 \
  sampling.add_identity_pairs=true \
  sampling.use_resampling=true \
  sampling.precursor_mass_mode=theoretical \
  sampling.precursor_noise_mode=mist_cf \
  sampling.train_exclude_smiles_file="${EXCLUDE_FILE}" \
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
  model.tasks.mces_bucket.use_mlp=true

echo "===== Training complete: $(date) ====="
