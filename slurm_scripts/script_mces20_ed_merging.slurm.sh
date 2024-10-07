#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -p zen2

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

srun python merge_edit_distance_mces_20_v2.py
