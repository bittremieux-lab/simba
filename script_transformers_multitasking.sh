#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1 --cpus-per-task=20

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 
#srun python training_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_precursor_augmentation_loss2--USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1

#srun python training_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_context500 --TRANSFORMER_CONTEXT=500   --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001
#srun python training_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_augmentations --TRANSFORMER_CONTEXT=100   --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001
#srun python training_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_augmentations_p20 --TRANSFORMER_CONTEXT=100   --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001

#srun python training_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_context200 --TRANSFORMER_CONTEXT=200   --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001
#srun python training_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_augmentations_p20_noprecursornegative --TRANSFORMER_CONTEXT=100   --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001
srun python training_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_augmentations_selective_ed --TRANSFORMER_CONTEXT=100   --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001


#srun python training_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_2024115 --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=0 --USE_EDIT_DISTANCE_REGRESSION=0 
#--enable_progress_bar=0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_newdata_20240925/ --extra_info=_multitasking_mces20raw_newdata
#--EDIT_DISTANCE_USE_GUMBEL=1
#--D_MODEL=128  
#--PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_multitasking_min_peaks/ --extra_info=_min_peaks
#srun python inference_multitasking.py --enable_progress_bar=0  --D_MODEL=128  
