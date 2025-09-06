#!/bin/bash
#SBATCH -p longq        # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -J train        # Gonderilen isin ismi
#SBATCH -o err-out/train-%j.out # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1    # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH --time=3-00:00:00
#SBATCH -N 1            # Gorev kac node'da calisacak?
#SBATCH -n 1            # Ayni gorevden kac adet calistirilacak?
#SBATCH --error=err-out/train-%j.err
#SBATCH --cpus-per-task 4  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.

echo $CUDA_VISIBLE_DEVICES
# module load gcc/12.1 # TRUBA'da gcc 9.8
# module load cuda/cuda-12.1-a100q
# Search modules at: https://www.uhem.itu.edu.tr/modules.html
# module çalışmıyor

nvidia-smi
echo $HOME
which nvcc

# source_dir="$HOME/Coref/err-out"
# destination_dir="$HOME/Coref/old_files/err-out"
# latest_files=($(ls -1 "$source_dir" | grep -E "[0-9]+" | sort -n | tail -2))

# for file in "$source_dir"/*; do
#     if [[ ! " ${latest_files[@]} " =~ " $(basename "$file") " ]]; then
#         mv "$file" "$destination_dir"
#     fi
# done

eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/unsloth_env/lib
conda activate unsloth_env
which python
conda info

XFORMERS_MORE_DETAILS=1 python ../../src/training/train_unified.py \
    --model_name "unsloth/llama-3-8b-Instruct-bnb-4bit" \
    --dataset_name "HFDS_train_few" \
    --ins_num 1 \
    --seed 0 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 4 \
    --resume_from_checkpoint
# CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 TORCHELASTIC_ERROR_FILE='/path/to/your/Coref/err-out/torch.out' accelerate launch --config_file /path/to/your/.cache/huggingface/accelerate/default_config.yaml train.py

# python load_and_save_model.py
# MAX_JOBS=8 pip install flash-attn --no-build-isolation # V100 uyumlu olmadığı için fp16 ve flash-attn yok

conda deactivate
