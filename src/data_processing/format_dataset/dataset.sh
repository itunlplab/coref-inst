#!/bin/bash

RAW_DIR="data/raw/"
DATASET_DIR="data/dataset/"
WINDOW_SIZE=1850
STRIDE_SIZE=0

mkdir -p $RAW_DIR
mkdir -p $DATASET_DIR

# Download ConllU data to RAW_DIR if --download flag is specified
if [[ $* == *--download* ]]; then
    sh get.sh --$RAW_DIR # Version 1.2
fi

# Create folders
for d in ${RAW_DIR}*/; do
    folder=$(basename $d)
    mkdir -p $DATASET_DIR$folder
done

rest=-3
NUM_JOBS="$(($(nproc)+$rest))" # This uses all processing units, you can change it.
echo "Out of $(nproc) processing units $NUM_JOBS are being used"

JOBS=0
# Transform every file and save to $DATASET_DIR
for p in ${RAW_DIR}*/*.conllu; do
    (
        file_name=$(basename "${p}" | cut -d. -f1)
        last_folder=$(basename "$(dirname "${p}")")
        output_path="$DATASET_DIR$last_folder/$file_name.jsonl"
        python3 dataset.py -f $file_name -p $p -o $output_path -w $WINDOW_SIZE -s $STRIDE_SIZE
    ) &

    # Manage parallel jobs
    JOBS=$((JOBS+1))
    if [ "$JOBS" -ge "$NUM_JOBS" ]; then
        wait -n  # Wait for any job to finish
        JOBS=$((JOBS-1))
    fi
done

# Wait for all background jobs to complete
wait

rm -r data/merged

# bash test_ds.sh
bash merge_files.sh data/dataset
# bash merge_files.sh data/test
python3 jsonl2HF.py
