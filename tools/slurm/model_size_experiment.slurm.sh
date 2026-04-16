#!/bin/bash
#
# Model-size benchmark for SimBA — 1 GPU, 1 node, accelgor.
#
# Pass N_LAYERS and D_MODEL at submit time:
#   sbatch --export=N_LAYERS=5,D_MODEL=256 tools/slurm/model_size_experiment.slurm.sh
#
# Full sweep (9 experiments):
#   for L in 2 5 8; do
#     for D in 128 256 512; do
#       sbatch --clusters=accelgor --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
#         --export=N_LAYERS=$L,D_MODEL=$D \
#         tools/slurm/model_size_experiment.slurm.sh
#     done
#   done

#SBATCH --job-name=simba_modelsize
#SBATCH --output=logs/model_size_%j_%N.out
#SBATCH --error=logs/model_size_%j_%N.err
#SBATCH -t 01:00:00
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
OUTPUT_DIR=${WORKDIR}/experiments/model_size/results
BATCH_SIZE=2048        # effective batch size; micro-batch auto-determined per model size
WARMUP_STEPS=50
MEASURE_STEPS=200
GPU_TYPE=A100
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

# Default values if not passed via --export
N_LAYERS=${N_LAYERS:-5}
D_MODEL=${D_MODEL:-256}

# ============================================================

cd "$WORKDIR"
mkdir -p logs experiments/model_size/results

source .venv/bin/activate

export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN

echo "======================================================"
echo "SimBA Model-Size Benchmark"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURM_NODELIST"
echo "  n_layers : $N_LAYERS"
echo "  d_model  : $D_MODEL"
echo "  GPU      : $GPU_TYPE"
echo "======================================================"

python tools/model_size_benchmark.py \
    --preprocessing-dir "$PREPROCESSING_DIR" \
    --effective-batch-size "$BATCH_SIZE" \
    --warmup-steps "$WARMUP_STEPS" \
    --measure-steps "$MEASURE_STEPS" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$OUTPUT_DIR" \
    --gpu-type "$GPU_TYPE" \
    --n-layers "$N_LAYERS" \
    --d-model "$D_MODEL"

echo "======================================================"
echo "Model-size benchmark finished: layers=${N_LAYERS} d_model=${D_MODEL}"
echo "Result: ${OUTPUT_DIR}/layers${N_LAYERS}_dmodel${D_MODEL}_${GPU_TYPE}_results.json"
echo "======================================================"
