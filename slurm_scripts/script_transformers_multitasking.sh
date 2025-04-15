#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p ampere_gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1 --cpus-per-task=20

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

nvidia-smi 

## Run gumbel
#srun python training_multitasking_generated_data.py --EDIT_DISTANCE_USE_GUMBEL=1 --enable_progress_bar=0  --D_MODEL=256  --ADD_HIGH_SIMILARITY_PAIRS=0 --extra_info=_generated_data_gumbel  --load_pretrained=1 --LR=0.00001

## Running 10 layers
#srun python training_multitasking_generated_data.py --enable_progress_bar=0  --N_LAYERS=10 --D_MODEL=128  --ADD_HIGH_SIMILARITY_PAIRS=0 --extra_info=_generated_data_20250206_no_pretraining  --load_pretrained=0 --LR=0.00001


## Running 5 layers
#srun python training_multitasking_generated_data.py --enable_progress_bar=0    --extra_info=_smooth_penalty_matrix  --load_pretrained=1 --pretrained_path=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/pretrained_model_256n/best_model.ckpt --LR=0.00001


python training_multitasking_generated_data.py --enable_progress_bar=0    --extra_info=_smooth_penalty_matrix  --load_pretrained=1 --pretrained_path=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/pretrained_model_256n/best_model.ckpt --LR=0.00001  --CHECKPOINT_DIR=/scratch/antwerpen/209/vsc20939/data/model_checkpoints/model_256n_smooth_penalty_matrix/ 

### Aditional commands
#srun python training_multitasking.py --enable_progress_bar=0   --extra_info=_multitasking_2024115 --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=0 --USE_EDIT_DISTANCE_REGRESSION=0 
#--enable_progress_bar=0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_newdata_20240925/ --extra_info=_multitasking_mces20raw_newdata
#--EDIT_DISTANCE_USE_GUMBEL=1
#--D_MODEL=128  
#--PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_multitasking_min_peaks/ --extra_info=_min_peaks
#srun python inference_multitasking.py --enable_progress_bar=0  --D_MODEL=128  
#python training_multitasking_generated_data_resampling.py --enable_progress_bar=0  --USE_RESAMPLING=1  --ADD_HIGH_SIMILARITY_PAIRS=0 --extra_info=_generated_data_resampling --USE_MCES20_LOG_LOSS=1 --use_cosine_distance=1 --USE_LOSS_WEIGHTS_SECOND_SIMILARITY=1 --load_pretrained=1 --LR=0.00001
