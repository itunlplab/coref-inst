# Coreference Resolution with Large Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Package Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/itunlplab/coref-inst)

This library provides an implementation for coreference resolution using fine-tuned large language models. The system converts coreference resolution into a text-to-text task where models learn to identify and group mentions that refer to the same entity.

## Overview

The project fine-tunes transformer models (Llama, Gemma, Mistral) on coreference resolution datasets using a novel mention masking approach. Models are trained to predict coreference chains by replacing masked mentions with appropriate cluster IDs.

## Installation

### From Source (Recommended for Development)

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/itunlplab/coref-inst.git
cd coref-inst

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install with Optional Dependencies

```bash
# For development and testing
pip install -e ".[dev]"

# For notebooks and visualization  
pip install -e ".[notebooks]"

# For evaluation tools
pip install -e ".[evaluation]"

# Install everything
pip install -e ".[dev,notebooks,evaluation]"
```

## Quick Start

### Command Line Interface

After installation, you can use the following command-line tools:

```bash
# Train a model
coref-train --model_name unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
           --dataset_name HFDS_5-shot --ins_num 1 --seed 42

# Run inference
coref-infer --model llama --instruction_number 1 \
           --dataset_dir HFDS_5-shot --seed 42 \
           --output_fp results.jsonl --model_dir ./saved_model \
           --ds_partition valid

# Process data
coref-process-data --input data.conllu --output data.jsonl

# Map predictions
coref-map-predictions --gold_conllu dev.conllu --llm_output results.jsonl \
                     --output_path output.conllu
```

### Python API

```python
from src.training.train_unified import main as train_model
from src.inference.infer import main as run_inference
from src.constants import LANGUAGE_DICT, MODEL_CONFIGS

# Use the library programmatically
# Training and inference can be controlled via Python scripts
```

## Core Components

### Data Processing Pipeline

#### `format_dataset/dataset.py`
Converts CoNLL-U format coreference data into training-ready JSONL format.

**Key Parameters:**
- `-f, --file_name`: Source filename for reference
- `-p, --path`: Input CoNLL-U file path  
- `-o, --output_path`: Output JSONL file path
- `-w, --window_size`: Context window size in tokens (default: 1500)
- `-s, --stride`: Sentence stride for sliding windows (default: 20)

**Features:**
- Dynamic document windowing based on tokenizer lengths
- Mention tag formatting: `<m>...</m>#MASK` for coreference spans
- Handles multiword tokens and empty mentions
- Fragments long sentences automatically

#### `format_dataset/dataset.sh`
Orchestrates the full dataset preparation pipeline:
- Downloads CorefUD-1.2 dataset
- Processes all `.conllu` files in parallel
- Merges processed files into HuggingFace dataset format

### Training Scripts

#### `train_unified.py`
Unified training script combining functionality from both few-shot and standard training approaches.

**Arguments:**
- `--model_name`: Base model (unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit, etc.)
- `--dataset_name`: Training dataset name (e.g., HFDS_5-shot)
- `--ins_num`: Instruction template number (1-6)
- `--seed`: Random seed for reproducibility

#### `train_few.sh` / `train_few_orgTraining.sh`
Batch training scripts that run experiments across:
- Multiple model architectures
- Different instruction numbers
- Various random seeds
- Configurable shot counts (1-5 shot learning)

### Prediction and Evaluation

#### `mapper/map_pred.py` 
Maps LLM predictions back to CoNLL-U format with proper coreference annotations.

**Key Parameters:**
- `-g, --gold_conllu`: Gold standard CoNLL-U file
- `-p, --donor_conllu`: Source CoNLL-U for mention positions  
- `-o, --output_path`: Output CoNLL-U file
- `-l, --llm_output_file`: LLM predictions in JSONL format
- `-f, --filename`: Filter by specific dataset
- `-d, --doc`: Process specific document ID

**Functionality:**
- Aligns LLM text outputs with original token positions
- Reconstructs coreference chains from cluster IDs
- Handles mention boundary alignment and multiword tokens
- Validates output format and computes evaluation scores

#### `infer.py`
Inference script for generating predictions on test data.

**Arguments:**
- `--model`: Model type (mistral, llama, gemma)
- `--instruction_number`: Which instruction template to use
- `--dataset_dir`: Dataset directory path
- `--seed`: Random seed
- `--output_fp`: Output file path
- `--model_dir`: Fine-tuned model directory

### Evaluation and Utilities

#### `auto_map.sh`
Automated evaluation script that:
- Runs inference across multiple documents
- Maps predictions to CoNLL-U format
- Computes CoNLL coreference scores
- Processes documents in parallel

#### Supporting Scripts
- `get.sh`: Downloads CorefUD dataset
- `submit_job.sh`: SLURM job submission with configurable resources
- `mapper.sh`: Standalone mapping script

## Instruction Templates

The system uses 6 different instruction templates (located in `instructions/`) for training and inference. Example from `instruction1.txt`:

```
The text is in English.
You are a coreference resolver.
Rewrite the sentence considering these rules:
Mentions are in <m>...</m>#MASK format.
Group the mentions that refer to same real-world entity.
If mentions refer to same thing write the same number instead of MASK.
If mentions represent different things write another number.
```

## Model Architecture Support

- **Llama 3.1 8B**: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- **Gemma 2 9B**: `unsloth/gemma-2-9b-it-bnb-4bit`
- **Mistral 7B**: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`

All models use 4-bit quantization via Unsloth for efficient training.

## Dataset Format

The system processes CorefUD format data and converts mentions using special tags:
- `<m>mention text</m>#ID`: Regular coreference mentions
- `</z>@ID`: Zero-width/empty mentions

Training examples use masking: `<m>mention</m>#MASK` -> `<m>mention</m>#0` (cluster ID)

## Usage

### Data Preparation
```bash
cd src/data_processing/format_dataset
bash dataset.sh --download
```

### Training
```bash
# Single experiment
python train_unified.py --model_name unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
                       --dataset_name HFDS_5-shot --ins_num 1 --seed 42

# Batch experiments
bash train_few.sh
```

### Inference and Evaluation
```bash
# Generate predictions
python infer.py --model llama --instruction_number 1 \
                --dataset_dir HFDS_5-shot --seed 42 \
                --output_fp results.jsonl --model_dir ./saved_model

# Map to CoNLL-U and evaluate
python mapper/map_pred.py -g dev.conllu -l results.jsonl -o output.conllu

# Automated evaluation
bash auto_map.sh
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Unsloth
- UDAPI
- PyYAML
- SLURM (for cluster computing, optional)

### Additional Tools Required

**CorefUD Scorer** (included as submodule):
```bash
# Initialize and update the submodule
git submodule update --init --recursive

# Install scorer dependencies  
cd corefud-scorer
pip install -r requirements.txt
```

**Note**: The CorefUD scorer is included as a git submodule for easier installation. If you cloned without `--recursive`, use the commands above to initialize it.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{corefinst_2025,
    author = {Pamay Arslan, Tuğba and Erol, Emircan and Eryiğit, Gülşen},
    title = {CorefInst: Leveraging LLMs for Multilingual Coreference Resolution},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {14},
    pages = {64-80},
    year = {2026},
    month = {01},
    issn = {2307-387X},
    doi = {10.1162/TACL.a.593},
    url = {https://doi.org/10.1162/TACL.a.593},
}
```

## Acknowledgments

This work was supported by the Scientific and Technological Research Council of Turkey (TÜBİTAK) under project grant No. 123E079 within the scope of the TÜBİTAK 2515 (European Cooperation in Science and Technology-COST) program, and was conducted within the framework of the ongoing UniDive COST Action (CA21167) on Universality, Diversity and Idiosyncrasy in Language Technology. The computing resources used in this work were provided by the National Center for High Performance Computing of Turkey (UHeM) under grant number 4021342024, and by the İTÜ Artificial Intelligence and Data Science Application and Research Center. We would also like to thank the anonymous reviewers and the action editors for their valuable feedback and constructive suggestions, which greatly improved the quality of this article.
