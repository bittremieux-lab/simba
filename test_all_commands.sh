#!/bin/bash

# Test script for all SIMBA commands
# Tests: preprocess, train, inference, analog-discovery
#
# Usage: bash test_all_commands.sh [DEVICE] [PRETRAINED_CHECKPOINT_DIR] [PRETRAINED_MODEL_NAME]
#   DEVICE: cpu or gpu (default: cpu)
#   PRETRAINED_CHECKPOINT_DIR: Path to pretrained model checkpoint directory (required for tests 5-6)
#   PRETRAINED_MODEL_NAME: Name of pretrained model file (required for tests 5-6)
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
    echo "Tests 5 and 6 (pretrained model tests) will be skipped."
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
echo "1/9 Testing: simba preprocess"
echo "--------------------------------"
uv run simba preprocess \
    preprocessing=fast_dev \
    paths.spectra_path=data/casmi2022.mgf \
    paths.preprocessing_dir=./test_full_workflow/preprocessed/

echo ""
echo "2/9 Testing: simba train"
echo "--------------------------------"
uv run simba train \
    training=fast_dev \
    paths.preprocessing_dir_train=./test_full_workflow/preprocessed/ \
    paths.checkpoint_dir=./test_full_workflow/checkpoints/ \
    training.epochs=3 \
    checkpoints.save_checkpoints=false \
    hardware.accelerator=$DEVICE

echo ""
echo "3/9 Testing: simba inference"
echo "--------------------------------"
uv run simba inference \
    inference=fast_dev \
    paths.checkpoint_dir=./test_full_workflow/checkpoints/ \
    paths.preprocessing_dir=./test_full_workflow/preprocessed/ \
    inference.preprocessing_pickle=mapping_unique_smiles.pkl \
    hardware.accelerator=$DEVICE

echo ""
echo "4/9 Testing: simba analog-discovery"
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
    echo "5/9 Testing: simba inference (pretrained model)"
    echo "--------------------------------"
    uv run simba inference \
        inference=fast_dev \
        paths.checkpoint_dir="$PRETRAINED_CHECKPOINT_DIR" \
        paths.preprocessing_dir=./test_full_workflow/preprocessed/ \
        inference.preprocessing_pickle=mapping_unique_smiles.pkl \
        hardware.accelerator=$DEVICE

    echo ""
    echo "6/9 Testing: simba analog-discovery (pretrained model)"
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
    echo "5/9 Skipping: simba inference (pretrained model) - no pretrained model provided"
    echo ""
    echo "6/9 Skipping: simba analog-discovery (pretrained model) - no pretrained model provided"
fi

echo ""
echo "7/9 Testing: simba train (with metadata features)"
echo "--------------------------------"
uv run simba train \
    training=fast_dev \
    paths.preprocessing_dir_train=./test_full_workflow/preprocessed/ \
    paths.checkpoint_dir=./test_full_workflow/checkpoints_metadata/ \
    model.features.use_ce=true \
    model.features.use_ion_activation=true \
    model.features.use_ion_method=true \
    training.epochs=3 \
    checkpoints.save_checkpoints=false \
    hardware.accelerator=$DEVICE

echo ""
echo "8/9 Testing: simba inference (with metadata features)"
echo "--------------------------------"
uv run simba inference \
    inference=fast_dev \
    paths.checkpoint_dir=./test_full_workflow/checkpoints_metadata/ \
    paths.preprocessing_dir=./test_full_workflow/preprocessed/ \
    inference.preprocessing_pickle=mapping_unique_smiles.pkl \
    model.features.use_ce=true \
    model.features.use_ion_activation=true \
    model.features.use_ion_method=true \
    hardware.accelerator=$DEVICE

echo ""
echo "9/9 Testing: simba analog-discovery (with metadata features)"
echo "--------------------------------"
uv run simba analog-discovery \
    analog_discovery=fast_dev \
    --model-path ./test_full_workflow/checkpoints_metadata/best_model.ckpt \
    --query-spectra data/casmi2022.mgf \
    --reference-spectra data/casmi2022.mgf \
    --output-dir ./test_full_workflow/analog_results_metadata/ \
    analog_discovery.query_index=0 \
    model.features.use_ce=true \
    model.features.use_ion_activation=true \
    model.features.use_ion_method=true \
    analog_discovery.device=$DEVICE

echo ""
echo "================================"
echo "✓ All commands completed successfully!"
echo "================================"
