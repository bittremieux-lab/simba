#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p pascal_gpu
#SBATCH --gpus=1
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers
python training.py --D_MODEL $D_MODEL --N_LAYERS $N_LAYERS
