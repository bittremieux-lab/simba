#!/bin/bash
#SBATCH -J simba_infer_msg_official
#SBATCH -p one_hour
#SBATCH --nodelist=asimov2
#SBATCH --gpus=nvidia_h200_nvl_4g.71gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /home/nkubrakov/simba/logs/simba_infer_msg_official_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_infer_msg_official_%j.err

# Inference on official MassSpecGym test split using best checkpoint.
# Outputs test_predictions.csv with smiles_0, smiles_1, mces_true, mces_pred.

set -uo pipefail

echo "===== SIMBA Inference: MSG official test split ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "Branch: $(git -C /home/nkubrakov/simba rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git -C /home/nkubrakov/simba rev-parse --short HEAD)"

PREPRO_DIR=/mnt/data2/nkubrakov/massspecgym/preprocessing_msg_official
CHECKPOINT_DIR=/mnt/data2/nkubrakov/experiments_3_dataset/training/msg_official_splits

cd /home/nkubrakov/simba

uv run simba inference \
  paths.preprocessing_dir="${PREPRO_DIR}" \
  paths.preprocessing_dir_train="${PREPRO_DIR}" \
  paths.preprocessing_pickle_file=mapping.pkl \
  paths.checkpoint_dir="${CHECKPOINT_DIR}" \
  inference.use_last_model=false \
  checkpoints.best_model_name="best_model-v1.ckpt" \
  inference.preprocessing_pickle=mapping.pkl \
  inference.batch_size=512 \
  inference.uniformize_testing=false \
  model.tasks.edit_distance.enabled=false \
  model.features.use_adduct=true \
  model.features.use_ce=true \
  model.features.use_ion_mode=true \
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=8 \
  logging.enable_progress_bar=false

echo "===== Inference complete: $(date) ====="
echo "CSV saved to: ${CHECKPOINT_DIR}/test_predictions.csv"
