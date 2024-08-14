#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1 --cpus-per-task=20

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 
srun python training_multitasking.py --enable_progress_bar=0 --D_MODEL=128  
#srun python inference_multitasking.py --enable_progress_bar=0  --D_MODEL=128  
