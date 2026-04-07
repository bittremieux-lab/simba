#!/bin/bash

#SBATCH --job-name=simba_scaling
#SBATCH --output=logs/scaling_%j_%N.out
#SBATCH --error=logs/scaling_%j_%N.err
#SBATCH -t 00:30:00
#SBATCH -p accelgor
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G

set -e

module swap cluster/accelgor

WORKDIR=/scratch/gent/vo/000/gvo00017/vsc21162/simba
PREPROCESSING_DIR=${WORKDIR}/preprocessed_massspecgym_22k_speedup
OUTPUT_DIR=${WORKDIR}/experiments/scaling_efficiency/results
BATCH_SIZE=2048
BASE_WARMUP_STEPS=150   # warmup steps for 1 GPU; scales inversely with GPU count
BASE_MEASURE_STEPS=1500 # measure steps for 1 GPU; scales inversely with GPU count
GPU_TYPE=A100
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}
# ============================================================

cd "$WORKDIR"
mkdir -p logs experiments/scaling_efficiency/results

# Activate virtual environment
source .venv/bin/activate

# ---- Multi-node communication setup ----
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN

NUM_NODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE:-1}
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# Scale steps inversely so total samples processed is constant across configs
WARMUP_STEPS=$(( BASE_WARMUP_STEPS / TOTAL_GPUS ))
MEASURE_STEPS=$(( BASE_MEASURE_STEPS / TOTAL_GPUS ))
# Ensure at least 50 steps even for large GPU counts
[[ $WARMUP_STEPS -lt 50 ]] && WARMUP_STEPS=50
[[ $MEASURE_STEPS -lt 50 ]] && MEASURE_STEPS=50

echo "======================================================"
echo "SimBA Scaling Benchmark"
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Nodes         : $NUM_NODES"
echo "  GPUs/node     : $GPUS_PER_NODE"
echo "  Total GPUs    : $TOTAL_GPUS"
echo "  Master addr   : $MASTER_ADDR:$MASTER_PORT"
echo "  Workers/proc  : $NUM_WORKERS"
echo "======================================================"

# srun launches one task per GPU (ntasks-per-node tasks per node)
# PyTorch Lightning reads SLURM_LOCALID / SLURM_PROCID / SLURM_NNODES
# to configure DDP automatically via SLURMEnvironment
srun --label python tools/scaling_benchmark.py \
    --preprocessing-dir "$PREPROCESSING_DIR" \
    --batch-size "$BATCH_SIZE" \
    --warmup-steps "$WARMUP_STEPS" \
    --measure-steps "$MEASURE_STEPS" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$OUTPUT_DIR" \
    --gpus-per-node "$GPUS_PER_NODE" \
    --num-nodes "$NUM_NODES" \
    --gpu-type "$GPU_TYPE"

echo "======================================================"
echo "Benchmark job finished for $TOTAL_GPUS GPU(s)"
echo "Result: ${OUTPUT_DIR}/${TOTAL_GPUS}gpu_${NUM_NODES}node_results.json"
echo "======================================================"
