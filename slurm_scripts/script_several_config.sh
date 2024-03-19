# submit several configurations
#for epochs in 3; do
#for BATCH_SIZE in 128 256; do
#      for LR in 0.00099; do
#        for N_LAYERS in 10 20 5; do
#        for D_MODEL in  64 128 32; do
#        sbatch --export=D_MODEL=$D_MODEL,N_LAYERS=$N_LAYERS,BATCH_SIZE=$BATCH_SIZE,LR=$LR,epochs=$epochs script_flexible.slurm.sh
#      done
#    done
#  done
#done
#done

for load_pretrained in 0; do
for epochs in 100; do
for BATCH_SIZE in 128; do
      for LR in 0.00009; do
        for N_LAYERS in 5; do
        for D_MODEL in  256 512; do
        sbatch --export=D_MODEL=$D_MODEL,N_LAYERS=$N_LAYERS,BATCH_SIZE=$BATCH_SIZE,LR=$LR,epochs=$epochs,load_pretrained=$load_pretrained script_flexible.slurm.sh
      done
    done
  done
done
done
done