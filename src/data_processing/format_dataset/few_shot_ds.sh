#!/usr/bin/env bash

# 

OUTPUT_DIR="data/merged"
extension=""
mkdir -p $OUTPUT_DIR

merge_jsonl_files() {
    local folder="$1"
    local output="$2"
    
    # Check if the folder exists
    if [ -d "$folder" ]; then
        # Merge train files
        for file in "$folder"/*-train.jsonl; do
            echo head -n 1 "$file"
            head -n 5 "$file" >> "$output/train_few$extension.jsonl"
        done
        
        echo "Merged files in $output"
    else
        echo "Folder $folder does not exist."
    fi
}

# Check if folder argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi

# Get the parent directory path from command line argument
parent_dir="$1"

# Check if the parent directory exists
if [ ! -d "$parent_dir" ]; then
    echo "Parent directory $parent_dir does not exist."
    exit 1
fi

# Find all directories under the parent directory
while IFS= read -r -d '' folder; do
    # Ensure it's a directory
    if [ -d "$folder" ]; then
        echo "Merging files in folder: $folder"
        merge_jsonl_files "$folder" "$OUTPUT_DIR"
    fi
done < <(find "$parent_dir" -type d -print0)