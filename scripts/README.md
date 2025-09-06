# Scripts Directory

This directory contains shell scripts organized by functionality for the coreference resolution pipeline.

## Directory Structure

```
scripts/
├── training/           # Training-related scripts
├── inference/          # Inference and prediction scripts  
├── evaluation/         # Evaluation and mapping scripts
├── setup/             # Dataset setup and preparation
├── slurm/             # SLURM job submission scripts
└── README.md          # This file
```

## Training Scripts (`training/`)

### `train_few.sh`
Batch training script for few-shot experiments across multiple models and configurations.

**Usage:**
```bash
cd scripts/training/
bash train_few.sh
```

### `train_few_orgTraining.sh`
Original training script with specific configurations.

### `train_unsloth.sh` 
Training script using Unsloth optimization.

## Inference Scripts (`inference/`)

### `inference_unsloth.sh`
Run inference with Unsloth-optimized models.

**Usage:**
```bash
cd scripts/inference/
bash inference_unsloth.sh
```

## Evaluation Scripts (`evaluation/`)

### `mapper.sh`
Maps LLM predictions back to CoNLL-U format and evaluates performance across all languages.

**Usage:**
```bash
cd scripts/evaluation/
./mapper.sh --llm-output results.jsonl --model-name llama --shot-count 5
```

**Options:**
- `--llm-output`: Path to LLM output JSONL file (required)
- `--gold-dir`: Directory with gold standard files (default: ../../data/raw)
- `--output-dir`: Output directory (default: ../../results/mapped)
- `--partition`: Data partition dev/test (default: dev)
- `--model-name`: Model name for output naming (default: model)
- `--shot-count`: Shot count for naming (default: 5)

### `auto_map.sh`
Automated evaluation pipeline that runs inference and mapping.

### `auto_infer.sh`
Automated inference across multiple configurations.

## Setup Scripts (`setup/`)

### `get.sh`
Downloads the CorefUD dataset and sets up the data directory structure.

**Usage:**
```bash
cd scripts/setup/
bash get.sh
```

## SLURM Scripts (`slurm/`)

### `submit_job.sh`
Submit training jobs to SLURM scheduler with configurable resources.

**Usage:**
```bash
cd scripts/slurm/
sbatch submit_job.sh
```

## Configuration Notes

- All scripts use relative paths from their directory location
- Scripts assume the standard project directory structure
- Update paths in scripts if you modify the project structure
- Some scripts may require conda environment activation (commented out for generalization)

## Common Workflows

### 1. Full Experiment Pipeline
```bash
# 1. Setup data
cd scripts/setup/
bash get.sh

# 2. Train models  
cd ../training/
bash train_few.sh

# 3. Run inference
cd ../inference/
bash inference_unsloth.sh

# 4. Evaluate results
cd ../evaluation/
./mapper.sh --llm-output ../../results/inference.jsonl --model-name llama
```

### 2. Single Model Training
```bash
cd scripts/training/
# Edit train_few.sh to specify single model configuration
bash train_few.sh
```

### 3. Custom Evaluation
```bash
cd scripts/evaluation/
./mapper.sh \
  --llm-output ../../results/my_results.jsonl \
  --gold-dir ../../data/test \
  --output-dir ../../results/evaluation \
  --partition test \
  --model-name mistral \
  --shot-count 3
```

## Troubleshooting

### Path Issues
- Ensure you run scripts from their intended directory
- Check that relative paths point to correct locations
- Verify data and model directories exist

### Permission Issues
```bash
# Make scripts executable
chmod +x scripts/**/*.sh
```

### Environment Issues
- Activate appropriate conda environment before running scripts
- Ensure all required packages are installed
- Check Python and CUDA versions compatibility

## Customization

### Adding New Scripts
1. Create script in appropriate category directory
2. Follow naming convention: `action_description.sh`
3. Include usage documentation in script header
4. Update this README

### Modifying Existing Scripts  
1. Update paths to match your setup
2. Modify model/dataset configurations as needed
3. Test thoroughly before running full experiments
4. Document changes in script headers