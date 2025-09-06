#!/bin/bash
#
# Unsloth Inference Script
# Runs inference using Unsloth-optimized models
#
# Usage: Run from scripts/inference/ directory
#        ./inference_unsloth.sh [model] [instruction_number] [seed]
#

# Default configuration
MODEL=${1:-"llama"}
INSTRUCTION_NUMBER=${2:-1}
SEED=${3:-42}

# Paths (relative to scripts/inference/)
INFERENCE_SCRIPT="../../src/inference/infer.py"
MODEL_BASE_DIR="../../models"
DATASET_DIR="../../data/old_ds/HFDS_infer"
OUTPUT_DIR="../../results/inference"
DS_PARTITION="valid"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Running Unsloth inference..."
echo "Model: $MODEL"
echo "Instruction: $INSTRUCTION_NUMBER" 
echo "Seed: $SEED"
echo "Dataset: $DATASET_DIR"
echo ""

# Conda environment setup (uncomment and modify as needed)
# eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniforge3/envs/unsloth_env/lib
# source "/path/to/your/miniforge3/condabin/conda"
# conda activate unsloth_env

# Check if inference script exists
if [[ ! -f "$INFERENCE_SCRIPT" ]]; then
    echo "Error: Inference script not found: $INFERENCE_SCRIPT"
    exit 1
fi

# Check if dataset exists
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    echo "Available datasets:"
    ls -la ../../data/old_ds/ | grep HFDS
    exit 1
fi

# Construct model directory path
MODEL_DIR="${MODEL_BASE_DIR}/${MODEL}${INSTRUCTION_NUMBER}_${SEED}"

# Check if model directory exists
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Warning: Model directory not found: $MODEL_DIR"
    echo "Looking for alternative model directories..."
    find "$MODEL_BASE_DIR" -name "*${MODEL}*" -type d | head -5
    echo ""
    echo "Please specify the correct model directory or train the model first."
    exit 1
fi

# Output file path
OUTPUT_FP="${OUTPUT_DIR}/${MODEL}${INSTRUCTION_NUMBER}_${SEED}.jsonl"

echo "Model directory: $MODEL_DIR"
echo "Output file: $OUTPUT_FP"
echo ""

# Run inference
python "$INFERENCE_SCRIPT" \
    --model "$MODEL" \
    --model_dir "$MODEL_DIR" \
    --instruction_number $INSTRUCTION_NUMBER \
    --output_fp "$OUTPUT_FP" \
    --seed $SEED \
    --dataset_dir "$DATASET_DIR" \
    --ds_partition "$DS_PARTITION"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✓ Inference completed successfully!"
    echo "Results saved to: $OUTPUT_FP"
else
    echo ""
    echo "✗ Inference failed!"
    exit 1
fi
