#!/bin/bash
#SBATCH -J simba_train_014_2_iceberg_aug_p50
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/014_2_iceberg_aug_p50_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/014_2_iceberg_aug_p50_1gpu/%x_%j.err

# ICEBERG train-spectra-augmentation experiment (see
# NOTES_014_2_ICEBERG_AUGMENTATION.md): identical config to 014_2
# (tools/slurm/014_2_mces_bucket_mlp_1gpu.slurm.sh) in every respect EXCEPT
# sampling.iceberg_mgf_path (a separate MGF of 120,050 ICEBERG-predicted
# spectra -- 24,010 Gaetan-train molecules x 5 collision energies,
# 15/25/35/45/55 eV chosen with the user directly, ICEBERG's own msg_all
# checkpoints) and sampling.iceberg_spectra_prob=0.5 -- for every train
# molecule that has a matched synthetic spectrum, each training sample has
# a 50% chance of drawing it instead of a real spectrum
# (simba/core/data/datasets/multitask_dataset.py::_sample_spectrum_index).
# Molecules with no synthetic match always fall back to real, unaffected.
# val/test are completely untouched -- this only ever changes what gets
# fed into the TRAIN side of a pair, and only during training (val uses
# deterministic first/last spectrum selection regardless of this setting).
#
# Verified before launching: the weighted-sampling logic in isolation
# (toy df_smiles, confirmed ~50% synthetic draw rate when available, always
# real when not, and iceberg_spectra_prob=0.0 exactly reproduces the
# original unweighted random.choice(indexes) behavior -- backward
# compatible for every other experiment). ICEBERG generation itself
# (120,050 spectra, ~16 min GPU job) and MGF conversion both completed with
# 0 errors and exact expected counts, checked directly.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 014_2 + ICEBERG train-spectra augmentation (prob=0.5) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/014_2_iceberg_aug_p50_1gpu
ICEBERG_MGF=/sofia/projects/2026_053/simba_project/data/analog_discovery/iceberg_train_augmentation/synthetic_train.mgf

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
  model.tasks.cosine_similarity.head_mode=cosine_no_head \
  model.tasks.mces_bucket.enabled=true \
  model.tasks.mces_bucket.use_mlp=true \
  sampling.iceberg_mgf_path="${ICEBERG_MGF}" \
  sampling.iceberg_spectra_prob=0.5

echo "===== Training complete: $(date) ====="
