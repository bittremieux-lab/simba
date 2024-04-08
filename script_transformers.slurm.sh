#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 

srun python training.py --enable_progress_bar=0 --extra_info=_unique_smiles_10_million_v4 --D_MODEL=512 --N_LAYERS=5 --LR=0.00001
srun python inference.py --enable_progress_bar=0
