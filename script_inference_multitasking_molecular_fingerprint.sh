#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p pascal_gpu
#SBATCH --gpus=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 
srun python inference_multitasking_molecular_fingerprint.py --enable_progress_bar=0   --extra_info=_multitasking_MOLECULAR_FINGERPRINT --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=0 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=0 --USE_MOLECULAR_FINGERPRINTS=1
#srun python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_multitasking_20241031_simetric_penalty --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=0 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=0
#srun python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_multitasking_20241031_noweights2_relu_logmces --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=0 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=0
#--enable_progress_bar=0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_newdata_20240925/ --extra_info=_multitasking_mces20raw_newdata
#--enable_progress_bar=0  --EDIT_DISTANCE_USE_GUMBEL=1 #gumbel
#--enable_progress_bar=0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_newdata_20240925/ --extra_info=_multitasking_mces20raw_newdata #new data
#--enable_progress_bar=0  --EDIT_DISTANCE_USE_GUMBEL=1 #gumbel
#--PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_multitasking_min_peaks/ --extra_info=_min_peaks
