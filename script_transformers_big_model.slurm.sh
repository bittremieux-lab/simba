#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1
#SBATCH -o stdout_transformers_big_model.file

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 

srun python test_main.py
