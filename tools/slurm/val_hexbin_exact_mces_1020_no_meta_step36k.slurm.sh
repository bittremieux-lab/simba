#!/bin/bash
#SBATCH -J simba_inf_exact_mces_no_meta_36k
#SBATCH -p one_hour
#SBATCH --nodelist=asimov
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH -o /home/nkubrakov/simba/logs/simba_inf_exact_mces_no_meta_36k_%j.out
#SBATCH -e /home/nkubrakov/simba/logs/simba_inf_exact_mces_no_meta_36k_%j.err

set -uo pipefail

echo "===== Inference: exact MCES [10-20] · no metadata @ step 36k ====="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Date: $(date)"
nvidia-smi

PREPRO_DIR=/mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020
EXP_DIR=/mnt/data/nkubrakov/experiments_3_dataset/training/msg_exact_mces_1020_no_meta
CHECKPOINT="${EXP_DIR}/checkpoint-epoch=03-step=36000.ckpt"
OUTPUT_DIR="${EXP_DIR}/val_hexbin_step36k"

mkdir -p "$OUTPUT_DIR"

cd /home/nkubrakov/simba

uv run python tools/run_val_hexbin.py \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --prepro_dir "${PREPRO_DIR}" \
  --batch_size 3072 \
  --num_workers 8 \
  model.features.use_adduct=false \
  model.features.use_ce=false \
  model.features.use_ion_mode=false \
  model.multitasking.learnable=false \
  model.tasks.edit_distance.enabled=false \
  model.tasks.edit_distance.n_classes=11

uv run python tools/plot_val_hexbin_balanced.py --val_dir "${OUTPUT_DIR}"
uv run python tools/plot_val_hexbin_balanced.py --val_dir "${OUTPUT_DIR}" --mces_max 20

echo "===== Done: $(date) ====="
