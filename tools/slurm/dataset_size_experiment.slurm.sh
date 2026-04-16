#!/bin/bash
#
# Dataset-size scalability benchmark for SimBA — 1 GPU, 1 node, accelgor.
#
# Pass N_SAMPLES and TIME_LIMIT at submit time:
#   sbatch --clusters=accelgor --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
#     --time=00:30:00 --export=N_SAMPLES=1000000 \
#     tools/slurm/dataset_size_experiment.slurm.sh
#
# Full sweep:
#   declare -A TIMES=([1000000]=00:15:00 [2000000]=00:25:00 [4000000]=00:50:00 \
#                     [8000000]=01:40:00 [16000000]=03:15:00 [32000000]=06:00:00)
#   for N in 1000000 2000000 4000000 8000000 16000000 32000000; do
#     sbatch --clusters=accelgor --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
#       --time=${TIMES[$N]} --export=N_SAMPLES=$N \
#       tools/slurm/dataset_size_experiment.slurm.sh
#   done

#SBATCH --job-name=simba_datasize
#SBATCH --output=logs/dataset_size_%j_%N.out
#SBATCH --error=logs/dataset_size_%j_%N.err
#SBATCH -t 06:00:00
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
OUTPUT_DIR=${WORKDIR}/experiments/dataset_size/results
BATCH_SIZE=2048
GPU_TYPE=A100
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

N_SAMPLES=${N_SAMPLES:-1000000}

cd "$WORKDIR"
mkdir -p logs experiments/dataset_size/results

source .venv/bin/activate

export OMP_NUM_THREADS=1

echo "======================================================"
echo "SimBA Dataset-Size Benchmark"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURM_NODELIST"
echo "  N_SAMPLES  : $N_SAMPLES"
echo "  GPU        : $GPU_TYPE"
echo "======================================================"

python tools/dataset_size_benchmark.py \
    --preprocessing-dir "$PREPROCESSING_DIR" \
    --n-samples "$N_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$OUTPUT_DIR" \
    --gpu-type "$GPU_TYPE"

echo "======================================================"
echo "Dataset-size benchmark finished: n_samples=${N_SAMPLES}"
echo "======================================================"
