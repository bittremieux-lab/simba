for EDIT_DISTANCE_USE_GUMBEL in 1; do
for GUMBEL_REG_WEIGHT in 0.05 0.1 0.2 0.5; do
for TAU_GUMBEL_SOFTMAX in 10; do
for load_pretrained in 1; do
for epochs in 100; do
for BATCH_SIZE in 128; do
      for LR in 0.00001; do
        for N_LAYERS in 5; do
        for D_MODEL in  256; do
        sbatch --export=D_MODEL=$D_MODEL,N_LAYERS=$N_LAYERS,BATCH_SIZE=$BATCH_SIZE,LR=$LR,epochs=$epochs,load_pretrained=$load_pretrained,EDIT_DISTANCE_USE_GUMBEL=$EDIT_DISTANCE_USE_GUMBEL,TAU_GUMBEL_SOFTMAX=$TAU_GUMBEL_SOFTMAX,GUMBEL_REG_WEIGHT=$GUMBEL_REG_WEIGHT script_flexible.slurm.sh
      done
    done
  done
done
done
done
done
done
done