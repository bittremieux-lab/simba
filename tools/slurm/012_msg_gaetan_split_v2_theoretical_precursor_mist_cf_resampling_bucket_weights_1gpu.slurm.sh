#!/bin/bash
#SBATCH -J simba_train_012_bucket_mass_weights
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/012_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_bucket_weights_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/012_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_bucket_weights_1gpu/%x_%j.err

# Experiment 012: identical CLI config to experiment 011 (Gaetan-split-v2,
# theoretical precursor mass + MIST-CF/BUDDY noise, MCES-weighted resampling
# for training only, dedicated self-pair bucket) -- the change here is
# entirely inside simba/workflows/training.py's prepare_data
# (use_mces_sampling branch), not a new config override, so 011 (already
# running with the pre-change code loaded into its process) is unaffected
# and remains a clean "resampling + self-pair bucket only" comparison point.
# New in the weighting itself:
#
#   - MCES_SAMPLING_BUCKET_MULTIPLIERS: on top of the existing inverse-bin-
#     frequency weight, self-pairs (bucket "self (MCES=0)") get an extra 4x
#     multiplier; the 4 buckets right above them -- (0,2.5], (2.5,5],
#     (5,7.5], (7.5,10], i.e. all of MCES<10 excluding exactly 0 -- get 2x;
#     everything MCES>10 is unchanged (1x).
#
#   - Mass-difference reweighting within each non-self MCES bucket: for
#     each bucket, the pairs at/below that bucket's own 10th-percentile
#     molecule mass difference (RDKit ExactMolWt) collectively get half
#     that bucket's sampling probability mass, and the rest of the bucket
#     gets the other half -- regardless of how lopsided the actual pair-
#     count split is (typically ~1:9). Self-pairs are always mass_diff==0
#     by construction, so this step is skipped for that bucket -- only the
#     4x multiplier above applies to it.
#
# Both steps combine multiplicatively with the base inverse-frequency
# weight, renormalized once at the end. See
# experiments/mass_diff_by_mces_bucket_gaetan_split_v2.png (built from
# tools/mass_diff_by_mces_bucket.py) for the mass-difference distributions
# these thresholds were chosen against.
#
# Unchanged from 011: Gaetan-split-v2 data, 1x H200, bs=2048, 32-true,
# lr=0.0001, 24 epochs, no MCES==20 exclusion, add_identity_pairs=true, no ED
# task, no learnable multitask weighting, no adduct/CE/ion-mode metadata,
# sampling.precursor_mass_mode=theoretical, sampling.precursor_noise_mode=mist_cf,
# sampling.use_resampling=true (training only -- validation always scores the
# full set, unweighted).

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 012: Gaetan split v2 · theoretical precursor mass + MIST-CF/BUDDY noise · MCES-weighted resampling (train only) w/ self-pair 4x + MCES<10 2x bucket multipliers + within-bucket mass-difference reweighting · cosine_no_head · 1x H200 · bs=2048 · 32-true · lr=0.0001 · 24 epochs ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/012_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_bucket_weights_1gpu

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
