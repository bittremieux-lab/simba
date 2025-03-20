#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -p zen2
#SBATCH --ntasks=10 --cpus-per-task=60
#SBATCH --exclusive  # Request exclusive access to the node

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

srun  --exclusive --ntasks=1 python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_0/ &
srun  --exclusive --ntasks=1  python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_1/ &
srun  --exclusive --ntasks=1 python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_2/ &
srun  --exclusive --ntasks=1 python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_3/ &
srun  --exclusive --ntasks=1 python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_4/ &
srun  --exclusive --ntasks=1 python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_5/ &
srun  --exclusive --ntasks=1 python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_6/ &
srun  --exclusive --ntasks=1 python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_7/ &
srun  --exclusive --ntasks=1 python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_8/ &
srun  --exclusive --ntasks=1  python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_9/ &
wait


