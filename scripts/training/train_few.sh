#!/bin/bash
#
# Few-Shot Training Script
# Trains models on few-shot coreference resolution datasets
#
# Usage: Run from scripts/training/ directory
#

# Configuration
MODELS=("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" "unsloth/gemma-2-9b-it-bnb-4bit" "unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
SEEDS=(0 42 199 285 404)
INSTRUCTION_NUMBERS=(1 2 3 4 5 6)
DATASET_NAME="HFDS_train_few"

# For quick testing, uncomment these:
# MODELS=("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
# SEEDS=(42)
# INSTRUCTION_NUMBERS=(1)

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Starting few-shot training..."

# Conda environment setup (uncomment and modify as needed)
# eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniforge3/envs/unsloth_env/lib
# source "/path/to/your/miniforge3/condabin/conda"
# conda activate unsloth_env

# Training script path (relative to scripts/training/)
TRAINING_SCRIPT="../../src/training/train_unified.py"

# Check if training script exists
if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    echo "Error: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Instructions: ${INSTRUCTION_NUMBERS[*]}"
echo "Dataset: $DATASET_NAME"
echo ""

for seed in "${SEEDS[@]}"; do
    for model in "${MODELS[@]}"; do
        for ins_num in "${INSTRUCTION_NUMBERS[@]}"; do
            echo "Training: Model=$model, Instruction=$ins_num, Seed=$seed"

            # Use srun if available (SLURM), otherwise run directly
            if command -v srun &> /dev/null; then
                XFORMERS_MORE_DETAILS=1 srun python "$TRAINING_SCRIPT" \
                    --model_name "$model" \
                    --dataset_name "$DATASET_NAME" \
                    --ins_num $ins_num \
                    --seed $seed
            else
                XFORMERS_MORE_DETAILS=1 python "$TRAINING_SCRIPT" \
                    --model_name "$model" \
                    --dataset_name "$DATASET_NAME" \
                    --ins_num $ins_num \
                    --seed $seed
            fi
            
            if [[ $? -eq 0 ]]; then
                echo "  ✓ Successfully trained: $model-$ins_num-$seed"
            else
                echo "  ✗ Failed training: $model-$ins_num-$seed"
            fi
            echo ""
        done
    done
done

echo "Few-shot training completed!"
echo "Models saved to: ../../models/"

# conda deactivate

conda deactivate






















# #!/bin/bash
# #SBATCH -p longq        # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
# #SBATCH -J train        # Gonderilen isin ismi
# #SBATCH -o err-out/train-%j.out # Ciktinin yazilacagi dosya adi
# #SBATCH --gres=gpu:1    # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
# #SBATCH -N 1            # Gorev kac node'da calisacak?
# #SBATCH -n 1            # Ayni gorevden kac adet calistirilacak?
# #SBATCH --error=err-out/train-%j.err
# #SBATCH --cpus-per-task 4  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.

# echo $CUDA_VISIBLE_DEVICES
# # module load gcc/12.1 # TRUBA'da gcc 9.8
# # module load cuda/cuda-12.1-a100q
# # Search modules at: https://www.uhem.itu.edu.tr/modules.html
# # module çalışmıyor

# nvidia-smi
# echo $HOME
# which nvcc

# # source_dir="$HOME/Coref/err-out"
# # destination_dir="$HOME/Coref/old_files/err-out"
# # latest_files=($(ls -1 "$source_dir" | grep -E "[0-9]+" | sort -n | tail -2))

# # for file in "$source_dir"/*; do
# #     if [[ ! " ${latest_files[@]} " =~ " $(basename "$file") " ]]; then
# #         mv "$file" "$destination_dir"
# #     fi
# # done

# eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/unsloth_env/lib
# conda activate unsloth_env
# which python
# conda info

# XFORMERS_MORE_DETAILS=1 python ../../src/training/train_unified.py
# # CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 TORCHELASTIC_ERROR_FILE='/path/to/your/Coref/err-out/torch.out' accelerate launch --config_file /path/to/your/.cache/huggingface/accelerate/default_config.yaml train.py

# # python load_and_save_model.py
# # MAX_JOBS=8 pip install flash-attn --no-build-isolation # V100 uyumlu olmadığı için fp16 ve flash-attn yok

# conda deactivate


# . "/path/to/your/miniforge3/etc/profile.d/mamba.sh"