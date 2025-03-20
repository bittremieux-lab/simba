#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 
srun python training.py --enable_progress_bar=0 --extra_info=_exhaustive_20240517 --D_MODEL=128 --LR=0.0001   --dataset_path=/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240516_exhaustive_cleaned.pkl
srun python inference.py --enable_progress_bar=0
