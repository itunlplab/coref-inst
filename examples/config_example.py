#!/usr/bin/env python3
"""
Example script demonstrating the configuration system.

This script shows how to:
1. Load default configuration from YAML
2. Override with command line arguments
3. Access configuration values
4. Use configuration-based training
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config_utils import get_config, setup_reproducibility
from src.constants import get_model_config, get_training_config, get_language_dict


def main():
    """Demonstrate configuration usage."""
    print("=== Coreference Resolution Configuration Example ===\n")
    
    # Load default configuration
    config = get_config()
    print("1. Loading default configuration from config/config.yaml")
    
    # Show some model configuration
    model_config = get_model_config()
    print(f"   Default model max_seq_length: {model_config.get('max_seq_length')}")
    print(f"   Default model dtype: {model_config.get('dtype')}")
    print(f"   Supported models: {list(model_config.get('supported_models', {}).keys())}")
    
    # Show training configuration
    training_config = get_training_config()
    print(f"   Default training epochs: {training_config.get('num_train_epochs')}")
    print(f"   Default batch size: {training_config.get('per_device_train_batch_size')}")
    print(f"   Default learning rate: {training_config.get('learning_rate')}")
    
    # Show language configuration
    languages = get_language_dict()
    print(f"   Supported languages: {len(languages)} languages")
    print(f"   Sample languages: {list(languages.items())[:3]}")
    
    print("\n2. Configuration access examples:")
    print(f"   config.get('model.max_seq_length'): {config.get('model.max_seq_length')}")
    print(f"   config.get('training.learning_rate'): {config.get('training.learning_rate')}")
    print(f"   config.get('nonexistent.key', 'default'): {config.get('nonexistent.key', 'default')}")
    
    print("\n3. Example: Override with simulated command line args")
    from argparse import Namespace
    
    # Simulate command line arguments
    args = Namespace(
        model_name='llama',
        learning_rate=1e-4,
        num_train_epochs=3,
        max_seq_length=4096
    )
    
    # Update config with args
    config.update_from_args(args)
    
    print(f"   After override - learning_rate: {config.get('training.learning_rate')}")
    print(f"   After override - epochs: {config.get('training.num_train_epochs')}")
    print(f"   After override - max_seq_length: {config.get('model.max_seq_length')}")
    
    print("\n4. Reproducibility setup")
    setup_reproducibility(config, seed=42)
    print("   ✓ Reproducibility configured with seed=42")
    
    print("\n5. Configuration sections available:")
    sections = ['model', 'training', 'data', 'inference', 'languages', 'wandb']
    for section in sections:
        if config.get(section):
            print(f"   ✓ {section}")
        else:
            print(f"   ✗ {section} (missing)")
    
    print("\n=== Configuration Example Complete ===")


if __name__ == "__main__":
    main()