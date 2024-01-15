#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p pascal_gpu
#SBATCH --gpus=1
#SBATCH -o stdout_pipeline.out

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate molecular_pairs
srun python compute_molecular_pairs.py --enable_progress_bar=0.0

source activate transformers
srun python training.py --enable_progress_bar=0
srun python inference.py 
