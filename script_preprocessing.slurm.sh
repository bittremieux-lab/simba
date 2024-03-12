#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -p zen2

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate molecular_pairs

srun python compute_molecular_pairs.py --enable_progress_bar=0.0
