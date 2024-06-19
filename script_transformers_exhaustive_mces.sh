#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 
srun python training_exhaustive_mces.py --enable_progress_bar=0 --extra_info=_mces_20240618 --D_MODEL=128 \
                --LR=0.0001 --load_pretrained=0 --dataset_path=/scratch/antwerpen/209/vsc20939/data/mces_neurips_nist.pkl\
                --bins_uniformise_TRAINING=6 --bins_uniformise_INFERENCE=6
#srun python inference.py --enable_progress_bar=0
