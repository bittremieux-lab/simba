#!/bin/bash
#SBATCH -J simba_infer_logmces_direct
#SBATCH -p one_day
#SBATCH --nodelist=asimov2
#SBATCH --gpus=nvidia_h200_nvl_2g.35gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o /home/nkubrakov/simba/logs/inference_logmces_direct_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/inference_logmces_direct_%j.err

set -uo pipefail

echo "===== SIMBA Inference — direct log-MCES head ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

PREPRO_DIR=/mnt/data2/nkubrakov/joint/preprocessing_scaffold_v1
CHECKPOINT_DIR=/mnt/data2/nkubrakov/experiments_3_dataset/training/metadata_adduct_ce_ionmode_log_mces_v5
OUTPUT_DIR=/mnt/data2/nkubrakov/experiments_3_dataset/metadata_adduct_ce_ionmode_log_mces_v5

mkdir -p "${OUTPUT_DIR}"
mkdir -p /home/nkubrakov/simba/logs

cd /home/nkubrakov/simba

uv run simba inference \
  paths.preprocessing_dir="${PREPRO_DIR}" \
  paths.preprocessing_dir_train="${PREPRO_DIR}" \
  paths.checkpoint_dir="${CHECKPOINT_DIR}" \
  paths.output_dir="${OUTPUT_DIR}" \
  inference.use_last_model=false \
  inference.preprocessing_pickle=mapping.pkl \
  inference.batch_size=3072 \
  inference.uniformize_testing=false \
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=4 \
  logging.enable_progress_bar=true \
  model.features.use_adduct=true \
  model.features.use_ce=true \
  model.features.use_ion_mode=true \
  model.tasks.mces.predict_log_mces_direct=true

echo "===== Inference complete: $(date) ====="
echo "Output: ${OUTPUT_DIR}"
