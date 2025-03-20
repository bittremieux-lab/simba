#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 

srun python training_with_low_range_pretraining.py --enable_progress_bar=0 --extra_info=_1_millions_low_range_pretrain_20240508 --dataset_path=/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240319_unique_smiles_1_million_v2_no_identity.pkl
srun python inference.py --enable_progress_bar=0 --extra_info=_1_millions_low_range_pretrain_20240508 --dataset_path=/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240319_unique_smiles_1_million_v2_no_identity.pkl
