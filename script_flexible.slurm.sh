#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers
python training_multitasking_generated_data.py --D_MODEL $D_MODEL --N_LAYERS $N_LAYERS --BATCH_SIZE $BATCH_SIZE  --LR $LR --epochs $epochs --load_pretrained $load_pretrained --enable_progress_bar=0 --extra_info=_hyperp 