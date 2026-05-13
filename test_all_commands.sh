#!/bin/bash

# Test script for all SIMBA commands
# Tests: preprocess, sweep (fresh+resume), train, inference, analog-discovery, molecular-network
#
# Usage: bash test_all_commands.sh [DEVICE] [PRETRAINED_CHECKPOINT_DIR] [PRETRAINED_MODEL_NAME]
#   DEVICE: cpu or gpu (default: cpu)
#   PRETRAINED_CHECKPOINT_DIR: Path to pretrained model checkpoint directory (required for tests 7-8)
#   PRETRAINED_MODEL_NAME: Name of pretrained model file (required for tests 7-8)
#
# Example: bash test_all_commands.sh gpu ./downl_data best_model.ckpt

set -e  # Exit on error

# Parse arguments
DEVICE="${1:-cpu}"
PRETRAINED_CHECKPOINT_DIR="${2}"
PRETRAINED_MODEL_NAME="${3}"

# Validate device argument
if [[ "$DEVICE" != "cpu" && "$DEVICE" != "gpu" ]]; then
    echo "Error: DEVICE must be 'cpu' or 'gpu', got: $DEVICE"
    exit 1
fi

# Check if pretrained model arguments are provided
if [[ -z "$PRETRAINED_CHECKPOINT_DIR" || -z "$PRETRAINED_MODEL_NAME" ]]; then
    echo "Warning: PRETRAINED_CHECKPOINT_DIR and PRETRAINED_MODEL_NAME not provided."
    echo "Tests 7 and 8 (pretrained model tests) will be skipped."
    SKIP_PRETRAINED=true
else
    SKIP_PRETRAINED=false
    PRETRAINED_MODEL_PATH="${PRETRAINED_CHECKPOINT_DIR}/${PRETRAINED_MODEL_NAME}"
    echo "Using pretrained model: $PRETRAINED_MODEL_PATH"
fi

echo "================================"
echo "Testing SIMBA CLI Commands"
echo "Device: $DEVICE"
echo "================================"

# Cleanup previous test runs
rm -rf test_full_workflow/
mkdir -p test_full_workflow

echo ""
echo "1/14 Testing: simba preprocess"
echo "--------------------------------"
uv run simba preprocess \
    preprocessing=fast_dev \
    paths.spectra_path=data/casmi2022.mgf \
    paths.preprocessing_dir=./test_full_workflow/preprocessed/

echo ""
echo "2/14 Testing: simba preprocess with cache (reuse distances)"
echo "------------------------------------------------------------"
uv run simba preprocess \
    preprocessing=fast_dev \
    paths.spectra_path=data/casmi2022.mgf \
    paths.preprocessing_dir=./test_full_workflow/preprocessed_cached/ \
    'preprocessing.precomputed_distances=[./test_full_workflow/preprocessed/]'

echo ""
echo "3/14 Testing: simba-sweep (fresh, 3 trials, 1 epoch)"
echo "------------------------------------------------------------"
uv run simba-sweep \
    +sweep=default \
    sweep.n_trials=3 \
    sweep.output_dir=./test_full_workflow/sweep_run1 \
    sweep.resume=false \
    paths.preprocessing_dir_train=./test_full_workflow/preprocessed/ \
    training.epochs=1 \
    hardware.accelerator=$DEVICE

echo ""
echo "4/14 Testing: simba-sweep (resume, 3 more trials with prior TPE knowledge)"
echo "------------------------------------------------------------"
uv run simba-sweep \
    +sweep=default \
    sweep.n_trials=3 \
    sweep.output_dir=./test_full_workflow/sweep_run1 \
    sweep.resume=true \
    paths.preprocessing_dir_train=./test_full_workflow/preprocessed/ \
    training.epochs=1 \
    hardware.accelerator=$DEVICE

echo ""
echo "5/14 Testing: simba train"
echo "--------------------------------"
uv run simba train \
    training=fast_dev \
    paths.preprocessing_dir_train=./test_full_workflow/preprocessed/ \
    paths.checkpoint_dir=./test_full_workflow/checkpoints/ \
    training.epochs=3 \
    checkpoints.save_checkpoints=false \
    hardware.accelerator=$DEVICE

echo ""
echo "6/14 Testing: simba inference"
echo "--------------------------------"
uv run simba inference \
    inference=fast_dev \
    paths.checkpoint_dir=./test_full_workflow/checkpoints/ \
    paths.preprocessing_dir=./test_full_workflow/preprocessed/ \
    inference.preprocessing_pickle=mapping_unique_smiles.pkl \
    hardware.accelerator=$DEVICE

echo ""
echo "7/14 Testing: simba analog-discovery"
echo "--------------------------------"
uv run simba analog-discovery \
    analog_discovery=fast_dev \
    --model-path ./test_full_workflow/checkpoints/best_model.ckpt \
    --query-spectra data/casmi2022.mgf \
    --reference-spectra data/casmi2022.mgf \
    --output-dir ./test_full_workflow/analog_results/ \
    analog_discovery.query_index=0 \
    analog_discovery.device=$DEVICE

if [[ "$SKIP_PRETRAINED" == "false" ]]; then
    echo ""
    echo "8/14 Testing: simba inference (pretrained model)"
    echo "--------------------------------"
    uv run simba inference \
        inference=fast_dev \
        paths.checkpoint_dir="$PRETRAINED_CHECKPOINT_DIR" \
        paths.preprocessing_dir=./test_full_workflow/preprocessed/ \
        inference.preprocessing_pickle=mapping_unique_smiles.pkl \
        hardware.accelerator=$DEVICE

    echo ""
    echo "9/14 Testing: simba analog-discovery (pretrained model)"
    echo "--------------------------------"
    uv run simba analog-discovery \
        analog_discovery=fast_dev \
        --model-path "$PRETRAINED_MODEL_PATH" \
        --query-spectra data/casmi2022.mgf \
        --reference-spectra data/casmi2022.mgf \
        --output-dir ./test_full_workflow/analog_results_pretrained/ \
        analog_discovery.query_index=0 \
        analog_discovery.device=$DEVICE
else
    echo ""
    echo "8/14 Skipping: simba inference (pretrained model) - no pretrained model provided"
    echo ""
    echo "9/14 Skipping: simba analog-discovery (pretrained model) - no pretrained model provided"
fi

echo ""
echo "10/14 Testing: simba train (with metadata features)"
echo "--------------------------------"
uv run simba train \
    training=fast_dev \
    paths.preprocessing_dir_train=./test_full_workflow/preprocessed/ \
    paths.checkpoint_dir=./test_full_workflow/checkpoints_metadata/ \
    model.features.use_adduct=true \
    model.features.use_ce=true \
    model.features.use_ion_activation=true \
    model.features.use_ion_method=true \
    training.epochs=3 \
    checkpoints.save_checkpoints=false \
    hardware.accelerator=$DEVICE

echo ""
echo "11/14 Testing: simba inference (with metadata features)"
echo "--------------------------------"
uv run simba inference \
    inference=fast_dev \
    paths.checkpoint_dir=./test_full_workflow/checkpoints_metadata/ \
    paths.preprocessing_dir=./test_full_workflow/preprocessed/ \
    inference.preprocessing_pickle=mapping_unique_smiles.pkl \
    model.features.use_adduct=true \
    model.features.use_ce=true \
    model.features.use_ion_activation=true \
    model.features.use_ion_method=true \
    hardware.accelerator=$DEVICE

echo ""
echo "12/14 Testing: simba analog-discovery (with metadata features)"
echo "--------------------------------"
uv run simba analog-discovery \
    analog_discovery=fast_dev \
    --model-path ./test_full_workflow/checkpoints_metadata/best_model.ckpt \
    --query-spectra data/casmi2022.mgf \
    --reference-spectra data/casmi2022.mgf \
    --output-dir ./test_full_workflow/analog_results_metadata/ \
    analog_discovery.query_index=0 \
    model.features.use_adduct=true \
    model.features.use_ce=true \
    model.features.use_ion_activation=true \
    model.features.use_ion_method=true \
    analog_discovery.device=$DEVICE

echo ""
echo "13/14 Testing: simba molecular-network (from scratch)"
echo "--------------------------------"
uv run simba molecular-network \
    --model-path ./test_full_workflow/checkpoints/best_model.ckpt \
    --input-spectra data/casmi2022.mgf \
    --output-dir ./test_full_workflow/molecular_network/ \
    molecular_network.device=$DEVICE \
    molecular_network.score_cutoff=0.0

echo ""
echo "14/14 Testing: simba molecular-network (reuse precomputed MCES)"
echo "--------------------------------"
uv run simba molecular-network \
    --model-path ./test_full_workflow/checkpoints/best_model.ckpt \
    --input-spectra data/casmi2022.mgf \
    --output-dir ./test_full_workflow/molecular_network_precomputed/ \
    molecular_network.device=$DEVICE \
    molecular_network.score_cutoff=0.0 \
    molecular_network.precomputed_mces=./test_full_workflow/molecular_network/similarity_mces.npy

echo ""
echo "================================"
echo "✓ All commands completed successfully!"
echo "================================"
