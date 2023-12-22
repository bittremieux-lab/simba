#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p zen2
#SBATCH -o stdout_transformers_fingerprints_2.file

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate molecular_pairs


srun python compute_fingerprints.py --enable_progress_bar=0 
