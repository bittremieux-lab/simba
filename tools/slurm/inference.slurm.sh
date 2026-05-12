#!/bin/bash
#SBATCH -J simba_inference_th20_asb
#SBATCH -p one_hour
#SBATCH --nodelist=asimov2
#SBATCH --gpus=nvidia_h200_nvl_4g.71gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o /home/nkubrakov/simba/logs/simba_inference_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_inference_%j.err

set -uo pipefail

echo "===== SIMBA Inference ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"

PREPRO_DIR=/mnt/data2/nkubrakov/massspecgym/preprocessing_th20_asb
CHECKPOINT_DIR=/mnt/data2/nkubrakov/massspecgym/checkpoints_th20_asb

cd /home/nkubrakov/simba
source .venv/bin/activate

simba inference \
  paths.preprocessing_dir="$PREPRO_DIR" \
  paths.preprocessing_dir_train="$PREPRO_DIR" \
  paths.preprocessing_pickle_file=mapping.pkl \
  paths.checkpoint_dir="$CHECKPOINT_DIR" \
  inference.use_last_model=false \
  inference.preprocessing_pickle=mapping.pkl \
  inference.batch_size=256 \
  inference.uniformize_testing=true \
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=4 \
  logging.enable_progress_bar=true

echo "===== Inference complete: $(date) ====="
