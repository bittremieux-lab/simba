
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
#SBATCH --output=/dev/null  # Redirect standard output to /dev/null
#SBATCH --error=/dev/null   # Redirect standard error to /dev/null

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate transformers

srun --exclusive --ntasks=1 python compute_molecular_pairs_mces.py --enable_progress_bar=0.0 --PREPROCESSING_DIR=/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_matching_ed_${i}/

EOF
done