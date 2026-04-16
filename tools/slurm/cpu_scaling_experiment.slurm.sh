#!/bin/bash
#
# CPU strong-scaling benchmark for SimBA on doduo.
#
# Submit once per core count, e.g.:
#   for N in 1 2 4 8 16 32; do
#     sbatch --clusters=doduo --cpus-per-task=$N tools/slurm/cpu_scaling_experiment.slurm.sh
#   done
#
# Results land in experiments/scaling_efficiency/results/
# Named: <N>cpu_1node_doduo_results.json

#SBATCH --job-name=simba_cpu_scaling
#SBATCH --output=logs/cpu_scaling_%j_%N.out
#SBATCH --error=logs/cpu_scaling_%j_%N.err
#SBATCH -t 04:00:00
#SBATCH -p doduo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1          # override with --cpus-per-task=N at submit time
#SBATCH --mem=32G                  # flat memory request; model + dataset needs ~10-15 GB regardless of core count

set -e

module swap cluster/doduo

WORKDIR=/scratch/gent/vo/000/gvo00017/vsc21162/simba
PREPROCESSING_DIR=${WORKDIR}/preprocessed_massspecgym_22k_speedup
OUTPUT_DIR=${WORKDIR}/experiments/scaling_efficiency/results
BATCH_SIZE=256           # smaller batch for CPU (memory / speed)
BASE_WARMUP_STEPS=5      # CPU training is ~100x slower; keep run time manageable
BASE_MEASURE_STEPS=20
GPU_TYPE=doduo           # used as label in output filename

# ============================================================

cd "$WORKDIR"
mkdir -p logs experiments/scaling_efficiency/results

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

NCORES=${SLURM_CPUS_PER_TASK:-1}

echo "======================================================"
echo "SimBA CPU Scaling Benchmark"
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Node          : $SLURM_NODELIST"
echo "  Cores         : $NCORES"
echo "  OMP threads   : $OMP_NUM_THREADS"
echo "======================================================"

python tools/scaling_benchmark.py \
    --preprocessing-dir "$PREPROCESSING_DIR" \
    --batch-size "$BATCH_SIZE" \
    --warmup-steps "$BASE_WARMUP_STEPS" \
    --measure-steps "$BASE_MEASURE_STEPS" \
    --num-workers 0 \
    --output-dir "$OUTPUT_DIR" \
    --gpus-per-node "$NCORES" \
    --num-nodes 1 \
    --gpu-type "$GPU_TYPE" \
    --accelerator cpu

echo "======================================================"
echo "CPU benchmark finished for $NCORES core(s)"
echo "Result: ${OUTPUT_DIR}/${NCORES}cpu_1node_${GPU_TYPE}_results.json"
echo "======================================================"
