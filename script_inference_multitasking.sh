#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p pascal_gpu
#SBATCH --gpus=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 
#srun python inference_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_noprecursor --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=0 --USE_PRECURSOR_MZ_FOR_MODEL=0
#srun python inference_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_precursor_augmentation_false_mz --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=0 
#srun python inference_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_precursor_augmentation_loss2 --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --BEST_MODEL_NAME=last.ckpt
#srun python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_contrastive --USE_MCES20_LOG_LOSS=0 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --BEST_MODEL_NAME=last.ckpt
#srun python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_multitasking_contrastive --USE_MCES20_LOG_LOSS=0 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --LR=0.00001 --BEST_MODEL_NAME=last.ckpt
#srun  python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_multitasking_maxp20_ed_improved --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001
srun  python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_multitasking_maxp40 --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001

#srun python inference_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_precursor_augmentation --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=0 
#srun python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_multitasking_precursor_randomized --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=0 
