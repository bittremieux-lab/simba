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


#srun  python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_generated_data_peak_dropout_more_data  --TRANSFORMER_CONTEXT=100  --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001 --BEST_MODEL_NAME=best_model-final_performance.ckpt
srun  python inference_multitasking.py --enable_progress_bar=0    --extra_info=_smooth_penalty_matrix  --load_pretrained=1 --pretrained_path=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/pretrained_model_256n/best_model.ckpt --LR=0.00001  --CHECKPOINT_DIR=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/model_256n_smooth_penalty_matrix/ 

#srun  python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_multitasking_augmentations  --TRANSFORMER_CONTEXT=100  --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001
#srun  python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_multitasking_context500  --TRANSFORMER_CONTEXT=500  --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001
#srun  python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_multitasking_augmentations_p20  --TRANSFORMER_CONTEXT=100  --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001
#srun  python inference_multitasking.py  --enable_progress_bar=0   --extra_info=_multitasking_augmentations_selective_ed --TRANSFORMER_CONTEXT=100   --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001
#srun  python inference_multitasking.py  --enable_progress_bar=0   --ADD_HIGH_SIMILARITY_PAIRS=0 --extra_info=_multitasking_nosim1 --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001
#srun  python inference_multitasking.py  --enable_progress_bar=0   --ADD_HIGH_SIMILARITY_PAIRS=0 --extra_info=_generated_data_logloss --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1  --load_pretrained=1 --LR=0.00001 
#srun  python inference_multitasking.py  --enable_progress_bar=0   --epochs=100 --ADD_HIGH_SIMILARITY_PAIRS=0 --extra_info=_several_config_10_0.5 --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1  --load_pretrained=1 --LR=0.00001 

#srun  python inference_multitasking.py  --enable_progress_bar=0   --ADD_HIGH_SIMILARITY_PAIRS=0 --extra_info=_multitasking_nosim1_nopretraining --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001

