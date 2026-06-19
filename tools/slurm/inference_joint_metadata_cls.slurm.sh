#!/bin/bash
#SBATCH -J simba_infer_joint_metadata_cls
#SBATCH -p one_day
#SBATCH --nodelist=asimov2
#SBATCH --gpus=nvidia_h200_nvl_2g.35gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o /home/nkubrakov/simba-integration/logs/inference_joint_metadata_cls_%j.out
#SBATCH -e /home/nkubrakov/simba-integration/logs/inference_joint_metadata_cls_%j.err

# Inference on joint dataset using model trained with adduct + CE + ion_mode metadata,
# CLS-pool encoder (metadata_adduct_ce_ionmode checkpoint).
# NOTE: current code uses pool="attention"; the AttnAggregator weights will be randomly
# initialized since this checkpoint was trained with pool=None. Transformer/metadata
# encoder weights load correctly. Results are approximate.
#
# Usage:
#   cd /home/nkubrakov/simba-integration
#   mkdir -p logs
#   sbatch tools/slurm/inference_joint_metadata_cls.slurm.sh

set -euo pipefail

echo "===== SIMBA Inference — Joint Dataset (metadata CLS-pool model) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

PREPRO_DIR=/mnt/data2/nkubrakov/joint/preprocessing_scaffold_v1
CHECKPOINT_DIR=/mnt/data2/nkubrakov/experiments_3_dataset/training/metadata_adduct_ce_ionmode
OUTPUT_DIR=/mnt/data2/nkubrakov/experiments_3_dataset/metadata_adduct_ce_ionmode

mkdir -p "${OUTPUT_DIR}"
mkdir -p /home/nkubrakov/simba-integration/logs

cd /home/nkubrakov/simba-integration || exit 1

uv run simba inference \
  paths.preprocessing_dir="${PREPRO_DIR}" \
  paths.preprocessing_dir_train="${PREPRO_DIR}" \
  paths.checkpoint_dir="${CHECKPOINT_DIR}" \
  paths.output_dir="${OUTPUT_DIR}" \
  inference.use_last_model=false \
  inference.preprocessing_pickle=mapping.pkl \
  inference.batch_size=3072 \
  inference.uniformize_testing=false \
  model.features.use_adduct=true \
  model.features.use_ce=true \
  model.features.use_ion_mode=true \
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=4 \
  logging.enable_progress_bar=true

echo "===== Inference complete: $(date) ====="
echo "Output: ${OUTPUT_DIR}"
