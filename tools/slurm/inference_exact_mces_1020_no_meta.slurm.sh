#!/bin/bash
#SBATCH -J simba_infer_exact_mces_no_meta
#SBATCH -p one_hour
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /home/nkubrakov/simba/logs/simba_infer_exact_mces_no_meta_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_infer_exact_mces_no_meta_%j.err

# Inference for job 8011: exact MCES [10-20] · no metadata · last checkpoint (step 36k).

set -uo pipefail

echo "===== SIMBA Inference: exact MCES no metadata · best checkpoint ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C /home/nkubrakov/simba rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C /home/nkubrakov/simba rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PREPRO_DIR=/mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020
CHECKPOINT_DIR=/mnt/data/nkubrakov/experiments_3_dataset/training/msg_exact_mces_1020_no_meta

cd /home/nkubrakov/simba

uv run simba inference \
  paths.preprocessing_dir="${PREPRO_DIR}" \
  paths.preprocessing_dir_train="${PREPRO_DIR}" \
  paths.preprocessing_pickle_file=mapping.pkl \
  paths.checkpoint_dir="${CHECKPOINT_DIR}" \
  inference.use_last_model=false \
  "checkpoints.best_model_name='checkpoint-epoch=03-step=36000.ckpt'" \
  inference.preprocessing_pickle=mapping.pkl \
  inference.batch_size=512 \
  inference.uniformize_testing=false \
  model.tasks.edit_distance.enabled=false \
  model.tasks.edit_distance.n_classes=11 \
  model.features.use_adduct=false \
  model.features.use_ce=false \
  model.features.use_ion_mode=false \
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=8 \
  logging.enable_progress_bar=false

echo "===== Inference complete: $(date) ====="
