#!/bin/bash
#
# Unified Training Script
# Runs training using the unified training script with configurable parameters
#
# Usage: Run from scripts/training/ directory
#        ./train_unified.sh [model] [dataset] [instruction_number] [seed] [additional_args...]
#

# Default configuration
MODEL=${1:-"llama"}
DATASET=${2:-"HFDS_train_few"}
INSTRUCTION_NUMBER=${3:-1}
SEED=${4:-42}

# Shift the first 4 arguments so $@ contains only additional arguments
shift 4

# Paths (relative to scripts/training/)
TRAINING_SCRIPT="../../src/training/train_unified.py"
CONFIG_FILE="../../config/config.yaml"

# Check if training script exists
if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    echo "Error: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

echo "Running unified training script..."
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Instruction: $INSTRUCTION_NUMBER" 
echo "Seed: $SEED"
echo "Additional arguments: $@"
echo ""

# Conda environment setup (uncomment and modify as needed)
# eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniforge3/envs/unsloth_env/lib
# source "/path/to/your/miniforge3/condabin/conda"
# conda activate unsloth_env

# Run training with all parameters
python "$TRAINING_SCRIPT" \
    --model_name "$MODEL" \
    --dataset_name "$DATASET" \
    --ins_num $INSTRUCTION_NUMBER \
    --seed $SEED \
    --config "$CONFIG_FILE" \
    "$@"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo "Model saved to: models/${MODEL}${INSTRUCTION_NUMBER}_${SEED}"
else
    echo ""
    echo "✗ Training failed!"
    exit 1
fi

# conda deactivate