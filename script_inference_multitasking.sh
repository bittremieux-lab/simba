#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p pascal_gpu
#SBATCH --gpus=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 
srun python inference_multitasking.py --enable_progress_bar=0 
#--PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_multitasking_min_peaks/ --extra_info=_min_peaks
