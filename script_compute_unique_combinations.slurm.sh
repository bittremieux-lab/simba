#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -p zen2
#SBATCH -o stdout_compute_unique_combinations.out

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate molecular_pairs

srun python compute_unique_combinations.py
