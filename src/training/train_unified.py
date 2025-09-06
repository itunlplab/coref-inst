import os
import sys
import argparse
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from transformers import TrainingArguments
    from unsloth import FastLanguageModel
    from datasets import DatasetDict
    from trl import SFTTrainer
    import wandb
    import torch

    from src.config_utils import get_config, setup_reproducibility, setup_wandb_from_config
    from src.constants import (
        get_language_dict, get_no_zero_languages, get_zero_mention_instruction,
        get_model_config, get_model_type
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    # For testing argument parsing
    def get_config(config_path=None):
        class MockConfig:
            def get(self, key, default=None):
                return default
            def update_from_args(self, args):
                pass
            def get_data_config(self):
                return {}
        return MockConfig()
    IMPORTS_AVAILABLE = False

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for model training configuration."""
    parser = argparse.ArgumentParser(description="Train a model with specified configurations.")
    
    # Load default config to get defaults
    config = get_config()
    
    # Required arguments
    parser.add_argument('--model_name', type=str, required=True, 
                        help="Name of the model to be trained (e.g., llama, mistral, gemma).")
    parser.add_argument('--dataset_name', type=str, required=True, 
                        help="Name of the dataset (e.g., HFDS_5-shot).")
    parser.add_argument('--ins_num', type=int, required=True, 
                        help="Which instruction to use (1-6).")
    parser.add_argument('--seed', type=int, required=True, 
                        help="Random seed for reproducibility.")
    
    # Optional arguments with config defaults
    parser.add_argument('--config', type=str, default=None,
                        help="Path to configuration YAML file.")
    parser.add_argument('--max_seq_length', type=int, 
                        default=config.get('model.max_seq_length', 8192),
                        help="Maximum sequence length for the model.")
    parser.add_argument('--num_train_epochs', type=int,
                        default=config.get('training.num_train_epochs', 5),
                        help="Number of training epochs.")
    parser.add_argument('--per_device_train_batch_size', type=int,
                        default=config.get('training.per_device_train_batch_size', 4),
                        help="Batch size per device during training.")
    parser.add_argument('--gradient_accumulation_steps', type=int,
                        default=config.get('training.gradient_accumulation_steps', 4),
                        help="Number of gradient accumulation steps.")
    parser.add_argument('--learning_rate', type=float,
                        default=config.get('training.learning_rate', 2e-4),
                        help="Learning rate for training.")
    parser.add_argument('--warmup_steps', type=int,
                        default=config.get('training.warmup_steps', 5),
                        help="Number of warmup steps.")
    parser.add_argument('--save_steps', type=int,
                        default=config.get('training.save_steps', 250),
                        help="Save checkpoint every X steps.")
    parser.add_argument('--logging_steps', type=int,
                        default=config.get('training.logging_steps', 1),
                        help="Log metrics every X steps.")
    parser.add_argument('--output_dir', type=str,
                        default=config.get('training.output_dir', 'models'),
                        help="Output directory for model and checkpoints.")
    parser.add_argument('--dataset_partition', type=str, 
                        choices=['train', 'valid', 'test'], default='valid',
                        help="Which dataset partition to use for training.")
    parser.add_argument('--shuffle', action='store_true',
                        help="Shuffle the training dataset.")
    parser.add_argument('--resume_from_checkpoint', action='store_true',
                        help="Resume training from the latest checkpoint.")
    parser.add_argument('--no_wandb', action='store_true',
                        help="Disable Weights & Biases logging.")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments first
    args = parse_arguments()
    
    if not IMPORTS_AVAILABLE:
        if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
            exit(0)  # Help was already shown
        print("Required dependencies not available. Please install unsloth, transformers, etc.")
        exit(1)

    # Load configuration and update with command line arguments
    config = get_config(args.config)
    config.update_from_args(args)

    # Extract configuration values
    model_name = args.model_name
    dataset_name = args.dataset_name
    ins_num = args.ins_num
    SEED = args.seed

    # Setup paths using config
    data_config = config.get_data_config()
    dataset_dir = os.path.join('..', '..', data_config.get('dataset_base_dir', 'data/old_ds'), dataset_name)

    # Setup reproducibility
    setup_reproducibility(config, SEED)

    # Get model type and supported models
    model_n = get_model_type(model_name)
    model_config = get_model_config(model_name)

def format_dataset(examples: Dict[str, Any]) -> Dict[str, Any]:
    """Format dataset examples into chat template format."""
    langs = examples["lang"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    zero_ins = get_zero_mention_instruction()

    for lang, inp, output in zip(langs, inputs, outputs):
        ins = INSTRUCTION
        if lang not in no_zero_lang:
            ins += " " + zero_ins
            
        if model_n == 'gemma':
            text = tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': ins + '\n\t\n' + inp},
                    {'role': 'assistant', 'content': output}
                ],
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            text = tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': ins},
                    {'role': 'user', 'content': inp},
                    {'role': 'assistant', 'content': output}
                ],
                tokenize=False,
                add_generation_prompt=False
            )
        texts.append(text)

    return {"text": texts}

def main():
    """Main entry point for training script."""
    global args, config, model_name, dataset_name, ins_num, SEED, model_n, model_config
    global full_model_name, dataset_dir, language_dict, no_zero_lang, INSTRUCTION
    global max_seq_length, dtype, load_in_4bit, trust_remote_code, model, tokenizer, trainer
    
    # Get full model path from config if it's a shorthand
    supported_models = model_config.get('supported_models', {})
    if model_name in supported_models:
        full_model_name = supported_models[model_name]
    else:
        full_model_name = model_name

    print(f"Using model: {full_model_name} (type: {model_n})")

    # Setup Weights & Biases if not disabled
    if not args.no_wandb:
        run_name = f"{model_n}-inst{ins_num}-SEED{SEED}"
        setup_wandb_from_config(config, run_name, group=model_n)

    # Save relevant files to wandb
    if not args.no_wandb:
        wandb.save("*.py")
        wandb.save('train_unified.sh')
        wandb.save('submit_job.sh')

    # Load model configuration from config
    max_seq_length = model_config.get('max_seq_length', 8192)
    dtype_str = model_config.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str) if hasattr(torch, dtype_str) else torch.float16
    load_in_4bit = model_config.get('load_in_4bit', True)
    trust_remote_code = model_config.get('trust_remote_code', True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=full_model_name,
        trust_remote_code=trust_remote_code,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # PEFT configuration from config
    peft_config = model_config.get('peft', {})
    model = FastLanguageModel.get_peft_model(
        model,
        r=peft_config.get('r', 16),
        target_modules=peft_config.get('target_modules', 
                                       ["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"]),
        lora_alpha=peft_config.get('lora_alpha', 16),
        lora_dropout=peft_config.get('lora_dropout', 0),
        bias=peft_config.get('bias', "none"),
        use_gradient_checkpointing=peft_config.get('use_gradient_checkpointing', "unsloth"),
        random_state=SEED,
        use_rslora=peft_config.get('use_rslora', False),
        loftq_config=peft_config.get('loftq_config', None),
    )

    # Load language configuration from config
    language_dict = get_language_dict()
    no_zero_lang = get_no_zero_languages()

    instruction_path = os.path.join('..', '..', 'config', 'instructions', f'instruction{ins_num}.txt')
    with open(instruction_path, 'r') as f:
        INSTRUCTION = f.read()

    dataset = DatasetDict.load_from_disk(dataset_dir)
    train_ds = dataset[args.dataset_partition]
    if args.shuffle:
        train_ds = train_ds.shuffle(seed=SEED)
    train_ds = train_ds.map(format_dataset, batched=True)

    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            num_train_epochs=args.num_train_epochs,
            report_to='wandb' if not args.no_wandb else None,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            lr_scheduler_type="constant",
            seed=SEED,
            output_dir=f"{args.output_dir}/{model_n}{ins_num}_{SEED}",
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=1,
        ),
    )
        
    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_model_path = f'{args.output_dir}/{model_n}{ins_num}_{SEED}'
    trainer.save_model(final_model_path)
    print(f"Training completed! Model saved to: {final_model_path}")
    return trainer_stats

if __name__ == "__main__":
    # If just testing help, allow it to work without all dependencies
    if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        args = parse_arguments()
        exit(0)
    elif IMPORTS_AVAILABLE:
        main()
    else:
        print("Required dependencies not available. Please install unsloth, transformers, etc.")
        exit(1)