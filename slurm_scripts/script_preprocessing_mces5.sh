#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -p zen2
#SBATCH --ntasks=1 --cpus-per-task=60
#SBATCH --exclusive  # Request exclusive access to the node

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

# run mces generation using threshold=5
srun python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --THRESHOLD_MCES=5.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold5_0/
