#!/bin/bash
#
# Setup Verification Script
# Verifies that all paths in scripts are correct and files exist
#
# Usage: Run from project root directory
#

echo "=== Coreference Resolution Project Setup Verification ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

error_count=0
warning_count=0

check_file() {
    local file="$1"
    local description="$2"
    
    if [[ -f "$file" ]]; then
        echo -e "  ${GREEN}✓${NC} $description: $file"
    else
        echo -e "  ${RED}✗${NC} $description: $file (NOT FOUND)"
        ((error_count++))
    fi
}

check_directory() {
    local dir="$1"
    local description="$2"
    
    if [[ -d "$dir" ]]; then
        echo -e "  ${GREEN}✓${NC} $description: $dir"
    else
        echo -e "  ${YELLOW}⚠${NC} $description: $dir (NOT FOUND)"
        ((warning_count++))
    fi
}

check_executable() {
    local script="$1"
    local description="$2"
    
    if [[ -x "$script" ]]; then
        echo -e "  ${GREEN}✓${NC} $description: $script (executable)"
    elif [[ -f "$script" ]]; then
        echo -e "  ${YELLOW}⚠${NC} $description: $script (not executable - run: chmod +x $script)"
        ((warning_count++))
    else
        echo -e "  ${RED}✗${NC} $description: $script (NOT FOUND)"
        ((error_count++))
    fi
}

echo "1. Checking Project Structure..."
check_directory "src" "Source code directory"
check_directory "config" "Configuration directory"
check_directory "data" "Data directory"
check_directory "scripts" "Scripts directory"
check_directory "models" "Models directory"
check_directory "results" "Results directory"

echo ""
echo "2. Checking Core Python Scripts..."
check_file "src/training/train_unified.py" "Training script"
check_file "src/inference/infer.py" "Inference script"
check_file "src/constants.py" "Constants module"
check_file "src/config_utils.py" "Configuration utilities"

echo ""
echo "3. Checking Configuration Files..."
check_file "config/config.yaml" "Main configuration"
check_file "config/requirements.txt" "Requirements file"
check_directory "config/instructions" "Instructions directory"

echo ""
echo "4. Checking Shell Scripts..."
check_executable "scripts/training/train_few.sh" "Training script"
check_executable "scripts/inference/inference_unsloth.sh" "Inference script"
check_executable "scripts/evaluation/mapper.sh" "Mapping script"
check_executable "scripts/evaluation/auto_infer.sh" "Auto inference script"
check_executable "scripts/evaluation/auto_map.sh" "Auto mapping script"
check_executable "scripts/setup/get.sh" "Dataset download script"

echo ""
echo "5. Checking Data Directories..."
check_directory "data/old_ds" "Legacy datasets directory"
if [[ -d "data/old_ds" ]]; then
    echo "    Available datasets:"
    ls -1 data/old_ds/ | sed 's/^/      /'
fi

echo ""
echo "6. Checking Package Structure..."
check_file "setup.py" "Setup script"
check_file "setup.cfg" "Setup configuration"
check_file "MANIFEST.in" "Package manifest"
check_file "LICENSE" "License file"
check_file "README.md" "README file"

echo ""
echo "7. Checking Documentation..."
check_file "CITATION.cff" "Citation file"
check_file "ETHICS.md" "Ethics statement"
check_file "PAPER_EXPERIMENTS.md" "Experiment documentation"
check_file "scripts/README.md" "Scripts documentation"

echo ""
echo "8. Testing Script Path References..."

# Check if scripts reference correct paths
echo "  Checking auto_infer.sh paths..."
if grep -q "../../src/inference/infer.py" scripts/evaluation/auto_infer.sh; then
    echo -e "    ${GREEN}✓${NC} Inference script path correct"
else
    echo -e "    ${RED}✗${NC} Inference script path incorrect"
    ((error_count++))
fi

if grep -q "../../data/old_ds/HFDS_infer" scripts/evaluation/auto_infer.sh; then
    echo -e "    ${GREEN}✓${NC} Dataset path correct"
else
    echo -e "    ${RED}✗${NC} Dataset path incorrect"
    ((error_count++))
fi

echo "  Checking train_few.sh paths..."
if grep -q "../../src/training/train_unified.py" scripts/training/train_few.sh; then
    echo -e "    ${GREEN}✓${NC} Training script path correct"
else
    echo -e "    ${RED}✗${NC} Training script path incorrect"
    ((error_count++))
fi

echo ""
echo "=== VERIFICATION SUMMARY ==="
if [[ $error_count -eq 0 && $warning_count -eq 0 ]]; then
    echo -e "${GREEN}✓ All checks passed! Project is properly configured.${NC}"
elif [[ $error_count -eq 0 ]]; then
    echo -e "${YELLOW}⚠ Setup complete with $warning_count warnings.${NC}"
    echo "  Address warnings to ensure full functionality."
else
    echo -e "${RED}✗ Setup has $error_count errors and $warning_count warnings.${NC}"
    echo "  Fix errors before proceeding."
fi

echo ""
echo "Quick start commands:"
echo "  1. Install package: pip install -e ."
echo "  2. Download data: cd scripts/setup && ./get.sh"
echo "  3. Train model: cd scripts/training && ./train_few.sh"
echo "  4. Run inference: cd scripts/inference && ./inference_unsloth.sh"
echo "  5. Evaluate: cd scripts/evaluation && ./mapper.sh --llm-output ../../results/inference/llama1_42.jsonl"

echo ""
exit $error_count