#!/bin/bash

# JOB_NAME="table2Replike_llama3_inference"
# file="inference_unsloth.sh"

# JOB_NAME="table2_inference_reRun"
# file="auto_map_tugba.sh"

# JOB_NAME="train"
# file="train_few.sh"

JOB_NAME="train-w_full_reproducibility"
file="train_few.sh"


MEMORY="64000"  # Memory in MB
GPUS="1"        # Number of GPUs
CPUS="8"        # CPUs per task
QUEUE="longq"   # Queue name
OUT_FILE="err-out/${JOB_NAME}-%j.out"
ERR_FILE="err-out/${JOB_NAME}-%j.err"

# Export variables and submit the job
sbatch \
    --mem=$MEMORY \
    --gres=gpu:$GPUS \
    --cpus-per-task=$CPUS \
    -p $QUEUE \
    -J $JOB_NAME \
    -o $OUT_FILE \
    --error=$ERR_FILE \
    $file
