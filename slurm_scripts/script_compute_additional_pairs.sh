#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -p zen2
#SBATCH -o stdout_compute_additional_pairs.out

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate molecular_pairs

srun python compute_molecular_pairs_additional_pairs.py --enable_progress_bar=0.0
