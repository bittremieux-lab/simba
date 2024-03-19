#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -p zen2
#SBATCH -o stdout_transformers_replicate_test_pairs.file

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate molecular_pairs


srun python replicate_test_pairs.py
