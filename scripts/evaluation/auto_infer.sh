#!/bin/bash
#
# Automated Inference Script
# Runs inference across multiple models, instruction templates, and seeds
#
# Usage: Run from project root or adjust paths accordingly
#

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "HOME: $HOME"
which nvcc

# Conda environment setup (uncomment and modify as needed)
# eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniforge3/envs/unsloth_env/lib
# source "/path/to/your/miniforge3/condabin/conda"
# conda activate unsloth_env

# Configuration
models=("llama" "gemma" "mistral")
SEEDS=(0 42 199 285 404)
output_dir="../../results/inference"
model_base_dir="../../models"
dataset_dir="../../data/old_ds/HFDS_infer"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

echo "Starting automated inference..."
echo "Models: ${models[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Output directory: $output_dir"
echo "Dataset directory: $dataset_dir"
echo ""

# Check if dataset directory exists
if [[ ! -d "$dataset_dir" ]]; then
    echo "Error: Dataset directory not found: $dataset_dir"
    echo "Available datasets:"
    ls -la ../../data/old_ds/ | grep HFDS
    exit 1
fi

for model in "${models[@]}"; do
    for ins_num in {1..5}; do
        for seed in ${SEEDS[@]}; do
            echo "Processing: $model, instruction $ins_num, seed $seed"
            
            output_fp="${output_dir}/${model}${ins_num}_${seed}.jsonl"
            model_dir_path="${model_base_dir}/${model}${ins_num}_few_seed${seed}"
            
            # Check if model directory exists
            if [[ ! -d "$model_dir_path" ]]; then
                echo "  Warning: Model directory not found: $model_dir_path"
                continue
            fi
            
            python ../../src/inference/infer.py \
                --model $model \
                --model_dir "$model_dir_path" \
                --instruction_number $ins_num \
                --output_fp "$output_fp" \
                --seed $seed \
                --dataset_dir "$dataset_dir" \
                --ds_partition "valid"
            
            if [[ $? -eq 0 ]]; then
                echo "  ✓ Successfully completed $model-$ins_num-$seed"
            else
                echo "  ✗ Failed $model-$ins_num-$seed"
            fi
        done
    done
done

echo ""
echo "Automated inference completed!"
echo "Results saved to: $output_dir"
echo "Generated files:"
ls -la "$output_dir"

# conda deactivate
