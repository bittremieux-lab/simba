#!/bin/bash
#SBATCH --job-name=simba-adam-baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=one_day
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/adam_baseline_%j.log
#SBATCH --error=logs/adam_baseline_%j.err

# SIMBA Training with Adam Optimizer (Baseline)
# Short comparison run with step-based validation and denser logging.

set -e

# Setup
cd /home/nkubrakov/simba
if command -v module >/dev/null 2>&1; then
    module load cuda/12.2 || true
else
    echo "module command not available; skipping module load"
fi

# Print environment info
echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU: $(nvidia-smi -L)"
else
    echo "GPU: nvidia-smi not available"
fi
echo "Start time: $(date)"
echo "========================"

# Create results directory
EXPERIMENT_DIR="experiments/results/adam_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_DIR"
mkdir -p logs
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "Experiment directory: $EXPERIMENT_DIR"
cp "$0" "$EXPERIMENT_DIR/$(basename "$0")"

# Time the training
echo "Starting training with Adam optimizer..."
SECONDS=0

uv run simba train \
    training.epochs=10 \
    training.val_check_interval=100 \
    optimizer.name=adam \
    optimizer.lr=1e-4 \
    checkpoints.save_checkpoints=false \
    paths.preprocessing_dir_train=./preprocessing_output/ \
    paths.checkpoint_dir="$CHECKPOINT_DIR" \
    hardware.accelerator=gpu \
    hardware.devices=1 \
    hardware.num_workers=16 \
    logging.log_every_n_steps=10 \
    2>&1 | tee "$EXPERIMENT_DIR/training.log"

ELAPSED=$SECONDS

# Save metadata
cat > "$EXPERIMENT_DIR/metadata.json" <<EOF
{
    "optimizer": "adam",
    "job_id": "$SLURM_JOB_ID",
    "node": "$SLURM_NODELIST",
    "training_time_seconds": $ELAPSED,
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "epochs": 10,
    "val_check_interval": 100,
    "log_every_n_steps": 10,
    "learning_rate": 1e-4,
    "checkpoint_dir": "$CHECKPOINT_DIR",
    "loss_history": "checkpoints/loss_history.json",
    "best_checkpoint": "checkpoints/best_model.ckpt"
}
EOF

echo "=== Training Complete ==="
echo "Total time: $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"
echo "Results saved to: $EXPERIMENT_DIR"
echo "Completion time: $(date)"
