# submit several configurations
for LR in 0.001 0.0001; do
for D_MODEL in 64 128 256; do
    for N_LAYERS in 5 8 10; do
      sbatch --export=D_MODEL=$D_MODEL,N_LAYERS=$N_LAYERS script_flexible.slurm.sh
    done
 done
done