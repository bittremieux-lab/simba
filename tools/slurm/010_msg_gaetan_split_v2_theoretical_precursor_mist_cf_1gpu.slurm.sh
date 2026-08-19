#!/bin/bash
#SBATCH -J simba_train_010_theoretical_precursor_mist_cf
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/010_msg_gaetan_split_v2_theoretical_precursor_mist_cf_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/010_msg_gaetan_split_v2_theoretical_precursor_mist_cf_1gpu/%x_%j.err

# Experiment 010: identical to experiment 009 (same split, same 1-GPU
# hyperparameters, same head_mode=cosine_no_head, same sampling flags), except
# for how the precursor-mass model input is derived, per Wout/Gaetan's Slack
# suggestion (see BASELINE_AND_DASHBOARD.md):
#
#   sampling.precursor_mass_mode=theoretical -- the base precursor mass is now
#   computed from each molecule's SMILES (RDKit ExactMolWt) + adduct, via
#   simba/core/chemistry/chem_utils.theoretical_precursor_mz, instead of being
#   read directly from the MGF's PEPMASS field (spec.precursor_mz). This
#   matters because MassSpecGym's own PRECURSOR_MZ field is itself already a
#   rounded theoretical value for the large majority of rows, not a real
#   instrument measurement -- so training on it directly (009's behavior)
#   partly trains on a noiseless, leakage-prone signal.
#
#   sampling.precursor_noise_mode=mist_cf -- replaces the training-time
#   perturbation applied on top of that base value. 009 (like all prior
#   experiments) used Sebastian's original scheme: uniform noise up to +/-1%
#   of precursor m/z, independently per side, 20% of the time an augmented
#   sample is drawn. This run instead uses a MIST-CF/BUDDY-style truncated
#   Gaussian: std = instrument-specific ppm tolerance / 5 (Orbitrap/FTICR=5ppm,
#   QTOF=10ppm, Ion Trap/Unknown=15ppm), applied at the same 20% rate. See
#   Augmentation.resample_precursor_masses_mist_cf in
#   simba/core/data/augmentation.py.
#
# Also folded into this run (found while implementing the above, not
# separately requested but fixed alongside it since they're on the same code
# path):
#   - ADDUCT_TO_MASS["[3M-H]-"] was sign-flipped (+1.007276 instead of
#     -1.007276), inconsistent with "[2M-H]-"/"[M-H]-"'s single-deprotonation
#     convention. Only affects [3M-H]- precursors, none of which are present
#     in MassSpecGym's MGF (only [M+H]+/[M+Na]+ occur there), so this had no
#     effect on 009 itself, but would have silently corrupted any future run
#     using that adduct.
#   - Augmentation.add_false_precursor_masses_positives (009's noise step)
#     overwrote precursor_mass_1 with a noised copy of precursor_mass_0's
#     value instead of precursor_mass_1's own value -- i.e. 009 trained on
#     pairs where, ~10% of the time (20% aug rate x 50% augmentation-call
#     rate), BOTH sides' "true" precursor mass were silently forced to the
#     same underlying value before independent noise was added. Fixed to use
#     each side's own precursor_mass_N as the base.
#
# Unchanged from 009 (see that script for the full rationale): Gaetan's v2
# split/preprocessing dir, head_mode=cosine_no_head, 1x H200, bs=2048,
# 32-true, lr=0.0001, 24 epochs, no MCES==20 exclusion, no resampling,
# full val, add_identity_pairs=true, no ED task, no learnable multitask
# weighting, no adduct/CE/ion-mode metadata.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 010: Gaetan split v2 · theoretical precursor mass + MIST-CF/BUDDY noise · cosine_no_head · 1x H200 · bs=2048 · 32-true · lr=0.0001 · 24 epochs · no MCES==20 exclusion · no resampling · full val ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/010_msg_gaetan_split_v2_theoretical_precursor_mist_cf_1gpu

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
  sampling.use_resampling=false \
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
