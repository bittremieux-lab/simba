#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p  broadwell


export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

srun python inference.py 
#srun python evaluate_model_outputs.py