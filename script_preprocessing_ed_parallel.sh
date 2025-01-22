## Computation of EDIT DISTANCE 
# Loop over index values from 0 to 9
for i in {0..9}
do
  # Replace the index in the directory path and submit the job with sbatch
  sbatch <<EOF
#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -p zen2
#SBATCH --ntasks=1 --cpus-per-task=60
#SBATCH --exclusive  # Request exclusive access to the node

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

srun python compute_molecular_pairs_mces.py --PREPROCESSING_NUM_WORKERS=30 --USE_EDIT_DISTANCE=1 --enable_progress_bar=0.0 --PREPROCESSING_CURRENT_NODE=${i}
EOF
done