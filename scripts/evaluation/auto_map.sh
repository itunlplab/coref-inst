#!/bin/bash
#
# Automated Mapping and Evaluation Script
# Maps inference results and evaluates coreference performance
#
# Usage: Run from scripts/evaluation/ directory
#

# Configuration
models=("llama" "gemma" "mistral")
docs=("GUM_textbook_labor" "GUM_letter_arendt" "GUM_fiction_lunre" "GUM_bio_emperor" "GUM_fiction_beast" "GUM_podcast_wrestling" "GUM_voyage_coron" "GUM_vlog_radiology" "GUM_voyage_athens")
seeds=(0 42 199 285 404)

# Directory paths (relative to scripts/evaluation/)
results_dir="../../results"
inference_dir="${results_dir}/inference"
mapping_dir="${results_dir}/mapping"
gold_data_dir="../../data/gold"
mapper_script="../../src/evaluation/mapper/map_pred.py"

# Create output directories
mkdir -p "$mapping_dir"
mkdir -p "${mapping_dir}/conllu"
mkdir -p "${mapping_dir}/eval"

echo "Starting automated mapping and evaluation..."
echo "Models: ${models[*]}"
echo "Documents: ${docs[*]}"
echo "Seeds: ${seeds[*]}"
echo ""

# Check if required directories exist
if [[ ! -d "$inference_dir" ]]; then
    echo "Error: Inference results directory not found: $inference_dir"
    echo "Run inference first using auto_infer.sh"
    exit 1
fi

# First loop: Process specific document mappings
echo "Processing document-specific mappings..."
for doc in "${docs[@]}"; do
    echo "Processing document: $doc"
    
    gold_file="${gold_data_dir}/dev_${doc}.conllu"
    inference_file="${inference_dir}/llama1_0.jsonl"  # Example file
    output_file="${mapping_dir}/conllu/llama3_${doc}-1.conllu"
    eval_file="${mapping_dir}/eval/llama3-1-${doc}.eval"
    
    # Check if gold file exists
    if [[ ! -f "$gold_file" ]]; then
        echo "  Warning: Gold file not found: $gold_file"
        continue
    fi
    
    # Check if inference file exists
    if [[ ! -f "$inference_file" ]]; then
        echo "  Warning: Inference file not found: $inference_file"
        continue
    fi
    
    # Run mapping
    python "$mapper_script" \
        --gold_conllu "$gold_file" \
        --output_path "$output_file" \
        --llm_output_file "$inference_file" \
        --doc "$doc" \
        2>&1 | tee "$eval_file"
    
    if [[ $? -eq 0 ]]; then
        echo "  ✓ Successfully mapped $doc"
    else
        echo "  ✗ Failed to map $doc"
    fi
done

echo ""
echo "Extracting CoNLL scores for all model combinations..."

# Second loop: Extract and display scores
for model in "${models[@]}"; do
    for num in {1..5}; do
        for seed in "${seeds[@]}"; do
            for doc in "${docs[@]}"; do
                echo "Checking: $model instruction=$num seed=$seed doc=$doc"

                eval_file="${mapping_dir}/eval/${model}${num}_${seed}_${doc}.eval"
                
                if [[ -f "$eval_file" ]]; then
                    score=$(grep "CoNLL score: " "$eval_file" | cut -d ":" -f 2 | tr -d ' ')
                    if [[ -n "$score" ]]; then
                        echo "  Score: $score"
                    else
                        echo "  No score found in $eval_file"
                    fi
                else
                    echo "  Evaluation file not found: $eval_file"
                fi
            done
        done
    done
done

echo ""
echo "Automated mapping and evaluation completed!"
echo "Results saved to: $mapping_dir"
echo ""
echo "Generated files:"
echo "CoNLL-U outputs: ${mapping_dir}/conllu/"
echo "Evaluation files: ${mapping_dir}/eval/"

