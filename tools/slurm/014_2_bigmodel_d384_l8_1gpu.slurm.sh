#!/bin/bash
#SBATCH -J simba_train_014_2_bigmodel_d384_l8
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/training/014_2_bigmodel_d384_l8_1gpu/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/training/014_2_bigmodel_d384_l8_1gpu/%x_%j.err

# Bigger-model experiment (see NOTES_014_2_BIGMODEL.md): identical config
# to 014_2 (tools/slurm/014_2_mces_bucket_mlp_1gpu.slurm.sh) in every
# respect EXCEPT model.transformer.d_model=384 (was 256) and
# model.transformer.n_layers=8 (was 5) -- roughly 3x the parameter count
# of every 013/014_x checkpoint so far (~4.6M params), all of which shared
# the same base architecture size and only varied the mces_bucket head's
# own internal toggles.
#
# Before launching this, fixed a real bug this would otherwise have hit
# silently: tools/simba_retrieval.py::load_model (and everything that calls
# it -- simba_retrieval_iceberg.py, analog_discovery_embed_rank.py) had
# d_model=256/n_layers=5 HARDCODED in the load_from_checkpoint() call.
# SimilarityModelMultitask never calls save_hyperparameters(), so the
# checkpoint itself doesn't record its own architecture size -- loading a
# bigger checkpoint through the old hardcoded path would have silently
# dropped every shape-mismatched weight (strict=False) instead of erroring,
# producing a mostly-random model with no warning. All three now take
# --d_model/--n_layers explicitly (default 256/5, unchanged for every
# existing checkpoint) -- MUST pass --d_model 384 --n_layers 8 when
# evaluating this checkpoint.
#
# Data/everything else identical to 014_2: Gaetan-split-v2, cosine_no_head,
# mces_bucket.use_mlp=true, 24 epochs, bs=2048, lr=0.0001.

set -uo pipefail

module load uv

SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba

echo "===== SIMBA Training 014_2 bigger model: d_model=384, n_layers=8 (was 256/5) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C "${SIMBA_DIR}" rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C "${SIMBA_DIR}" rev-parse --short HEAD)"
nvidia-smi

PREPRO_DIR=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2
MGF=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf
OUTPUT_DIR=/sofia/projects/2026_053/simba_project/experiments/training/014_2_bigmodel_d384_l8_1gpu

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
  model.transformer.d_model=384 \
  model.transformer.n_layers=8

echo "===== Training complete: $(date) ====="
