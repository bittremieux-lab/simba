#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1 --cpus-per-task=20

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 
srun python training_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_nologmces --USE_MCES20_LOG_LOSS=0
#--enable_progress_bar=0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_newdata_20240925/ --extra_info=_multitasking_mces20raw_newdata
#--EDIT_DISTANCE_USE_GUMBEL=1
#--D_MODEL=128  
#--PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_multitasking_min_peaks/ --extra_info=_min_peaks
#srun python inference_multitasking.py --enable_progress_bar=0  --D_MODEL=128  
