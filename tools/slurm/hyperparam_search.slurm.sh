#!/bin/bash
#SBATCH --job-name=simba_hyperparam
#SBATCH --output=logs/hyperparam_search_%j.out
#SBATCH --error=logs/hyperparam_search_%j.err
#SBATCH -t 24:00:00          # max for ampere_gpu; 10 trials x 10 epochs fits comfortably
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p litleo

# ============================================================
# SimBA Hyperparameter Search — 1 SLURM job, sequential Optuna trials
#
# All N_TRIALS run one-by-one inside this single job. The Optuna
# TPE sampler learns from each trial and picks smarter configs next.
# All results are persisted to an SQLite DB for restart support.
#
# RESTART: set RESUME=true and keep SWEEP_OUTPUT_DIR pointing to the
#   same folder. Optuna skips completed trials and continues.
# ============================================================

set -euo pipefail

cd /scratch/gent/vo/000/gvo00017/vsc21162/simba || exit 1
mkdir -p logs
source .venv/bin/activate

# ============================================================
# CONFIGURATION — edit these
# ============================================================
PREPROCESSING_DIR=./preprocessed_massspecgym_22k_speedup
SWEEP_OUTPUT_DIR=./sweeps/run_v1
N_TRIALS=15           # trials to run
EPOCHS=50             # epochs per trial (EarlyStopping may cut shorter)
EARLY_STOP_PATIENCE=5 # val_loss patience per trial; 0=disabled
RESUME=false
# ============================================================

DB="${SWEEP_OUTPUT_DIR}/trials.json"

if [ "$RESUME" = true ]; then
    if [ ! -f "${DB}" ]; then
        echo "ERROR: RESUME=true but no trials.json found in ${SWEEP_OUTPUT_DIR}"
        exit 1
    fi
    echo "Resuming existing study from ${DB}"
else
    # Fresh start: remove any stale trials.json from a previous failed attempt
    rm -f "${DB}"
fi

echo "======================================================"
echo "SimBA Hyperparameter Search"
echo "  Preprocessing dir : $PREPROCESSING_DIR"
echo "  Sweep output dir  : $SWEEP_OUTPUT_DIR"
echo "  Trials            : $N_TRIALS (sequential, 1 GPU)"
echo "  Epochs / trial    : $EPOCHS"
echo "  Resume            : $RESUME"
echo "  Trials file       : $DB"
echo "======================================================"

mkdir -p "$SWEEP_OUTPUT_DIR"

# All trials run sequentially in this job. simba-sweep owns the Optuna loop
# internally — no hydra-optuna-sweeper plugin, no SQLite, no SQLAlchemy.
# Results are written to trials.json after every trial.
simba-sweep \
    +sweep=default \
    sweep.n_trials=$N_TRIALS \
    sweep.output_dir=$SWEEP_OUTPUT_DIR \
    sweep.resume=$RESUME \
    sweep.study_name=simba_hyperparam_search \
    "paths.preprocessing_dir_train=$PREPROCESSING_DIR" \
    "training.epochs=$EPOCHS" \
    "training.early_stopping_patience=$EARLY_STOP_PATIENCE" \
    "hardware.accelerator=gpu"

echo "======================================================"
echo "Sweep done! $N_TRIALS trials completed."
echo "Results : cat ${DB} | python -m json.tool | grep value"
echo "Best    : python -c \"import json; t=json.load(open('${DB}')); b=min((x for x in t if x['status']=='completed'), key=lambda x: x['value']); print(b['params'], b['value'])\""
