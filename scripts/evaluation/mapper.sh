#!/bin/bash
#
# Coreference Resolution Mapping Script
# Maps LLM predictions back to CoNLL-U format and evaluates performance
#
# Usage: ./mapper.sh [OPTIONS]
#   --llm-output    Path to LLM output JSONL file (required)
#   --gold-dir      Directory containing gold standard CoNLL-U files (default: data/raw)
#   --output-dir    Output directory for mapped files (default: results/mapped)
#   --partition     Data partition to use: dev/test (default: dev)
#   --model-name    Model name for output naming (default: model)
#   --shot-count    Shot count for naming (default: 5)
#

# Default values
DEFAULT_GOLD_DIR="../../data/raw"
DEFAULT_OUTPUT_DIR="../../results/mapped"
DEFAULT_PARTITION="dev"
DEFAULT_MODEL_NAME="model"
DEFAULT_SHOT_COUNT="5"

# Parse command line arguments
GOLD_DIR=${DEFAULT_GOLD_DIR}
OUTPUT_DIR=${DEFAULT_OUTPUT_DIR}
PARTITION=${DEFAULT_PARTITION}
MODEL_NAME=${DEFAULT_MODEL_NAME}
SHOT_COUNT=${DEFAULT_SHOT_COUNT}
LLM_OUTPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --llm-output)
            LLM_OUTPUT="$2"
            shift 2
            ;;
        --gold-dir)
            GOLD_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --shot-count)
            SHOT_COUNT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --llm-output JSONL_FILE [OPTIONS]"
            echo "  --llm-output    Path to LLM output JSONL file (required)"
            echo "  --gold-dir      Directory containing gold standard CoNLL-U files"
            echo "  --output-dir    Output directory for mapped files"
            echo "  --partition     Data partition: dev/test (default: dev)"
            echo "  --model-name    Model name for output naming"
            echo "  --shot-count    Shot count for naming"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$LLM_OUTPUT" ]]; then
    echo "Error: --llm-output is required"
    echo "Use --help for usage information"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# CorefUD language codes
langs="ca_ancora cs_pcedt cs_pdt cu_proiel de_parcorfull de_potsdamcc en_gum en_litbank en_parcorfull es_ancora fr_democrat grc_proiel hbo_ptnk hu_korkor hu_szegedkoref lt_lcc no_bokmaalnarc no_nynorsknarc pl_pcc ru_rucor tr_itcc"

echo "Starting coreference mapping evaluation..."
echo "LLM Output: $LLM_OUTPUT"
echo "Gold Directory: $GOLD_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Partition: $PARTITION"
echo "Model: $MODEL_NAME (${SHOT_COUNT}-shot)"
echo ""

# Process each language
for lang in $langs; do
    echo "Processing $lang..."
    
    gold_file="$GOLD_DIR/$lang/$lang-corefud-$PARTITION.conllu"
    output_file="$OUTPUT_DIR/${lang}_${MODEL_NAME}_${SHOT_COUNT}shot.conllu"
    eval_file="$OUTPUT_DIR/${lang}_${MODEL_NAME}_${SHOT_COUNT}shot.eval"
    
    # Check if gold file exists
    if [[ ! -f "$gold_file" ]]; then
        echo "  Warning: Gold file not found: $gold_file"
        continue
    fi
    
    # Run mapping
    python ../../src/evaluation/mapper/map_pred.py \
        --gold_conllu "$gold_file" \
        --llm_output_file "$LLM_OUTPUT" \
        --output_path "$output_file" \
        --filename "$lang" \
        2>&1 | tee "$eval_file"
    
    if [[ $? -eq 0 ]]; then
        echo "  ✓ Successfully mapped $lang"
    else
        echo "  ✗ Failed to map $lang"
    fi
done

echo ""
echo "Mapping completed. Results in: $OUTPUT_DIR"
echo "Summary files:"
find "$OUTPUT_DIR" -name "*.eval" -exec basename {} \; | sort
