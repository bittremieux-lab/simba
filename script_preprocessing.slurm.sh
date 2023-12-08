#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1
#SBATCH -o stdout_transformers_preprocessing.file

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate molecular_pairs

nvidia-smi 

srun python mol_test.py
