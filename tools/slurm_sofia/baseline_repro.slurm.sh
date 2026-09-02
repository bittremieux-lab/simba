#!/bin/bash
#SBATCH -J simba_triplet_baseline_repro
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/simba-triplet-experiments/baseline_repro/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/simba-triplet-experiments/baseline_repro/%x_%j.err

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba-triplet
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/simba-triplet-experiments/baseline_repro

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf

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
  model.tasks.mces_bucket.enabled=true \
  model.tasks.mces_bucket.use_mlp=true

echo "Training complete: $(date)"
