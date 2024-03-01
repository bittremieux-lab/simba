#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p  ampere_gpu
#SBATCH --gpus=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

srun python inference.py --enable_progress_bar=0 --extra_info=_use_maldi_2
#srun python evaluate_model_outputs.py --enable_progress_bar=0
