# submit several configurations
for D_MODEL in 16 32; do
    for N_LAYERS in 2 5; do
      sbatch --export=D_MODEL=$D_MODEL,N_LAYERS=$N_LAYERS script_flexible.slurm.sh
    done
 done
