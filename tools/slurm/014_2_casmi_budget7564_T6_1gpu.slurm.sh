#!/bin/bash
#SBATCH -J simba_train_014_2_budget7564_T6
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/014_2_casmi_budget7564_T6_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/014_2_casmi_budget7564_T6_1gpu/%x_%j.err

# Budget-matched CASMI-distance sweep (see NOTES_014_2_ANALOG_DISCOVERY.md,
# "Budget-matched exclusion sets" -- addresses Wout's confound: the earlier
# excl{4,6,8,10,12,14} sweep varied training-set SIZE and distance-from-
# CASMI at the same time). Every condition in this sweep trains on exactly
# 7,564 molecules (matching threshold=14's own eligible count, the smallest
# in the original sweep) -- ONLY which specific molecules differs: among
# Gaetan-train molecules with min GT-MCES distance to any CASMI query >= T,
# the 7,564 CLOSEST to CASMI are kept (train_exclude_smiles_file excludes
# the other 16,446, same count for every T). T=6 (this file) means a
# minimum-distance floor of 6 on top of the closest-to-CASMI selection --
# distance range [6.0, 10.0] (verified when the exclusion
# file was built). This is NOT the same as the existing "014_2 baseline"
# analog-discovery row, which trains on all 24,010 molecules unrestricted;
# this T=0 is its own "closest possible, same budget as every other row
# here" anchor point.
#
# Differences from 014_2 itself: sampling.train_exclude_smiles_file (see
# above), training.val_check_interval=3000 (was 1000 -- less frequent
# checkpointing/validation overhead, no other reason). training.epochs is
# back to 014_2's own 24 (NOT the earlier sweep's 240) -- that 10x was
# compensating for a smaller-than-014_2 dataset with fewer pairs/epoch,
# which no longer applies now that every condition here has the identical
# molecule count, so the identical epoch budget is the fair comparison.
# Preprocessing data unmodified -- exclusion happens at data-load time.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 014_2 budget-matched sweep, T=6 (7,564 closest-to-CASMI train molecules, min_dist>=6) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/014_2_casmi_budget7564_T6_1gpu
EXCLUDE_FILE=/sofia/projects/2026_053/simba_project/data/analog_discovery/train_exclude_smiles_budget7564_T6.txt

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
  training.val_check_interval=3000 \
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
