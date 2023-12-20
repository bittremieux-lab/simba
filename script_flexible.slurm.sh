#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers
python training.py --d_model $D_MODEL --n_layers $N_LAYERS
