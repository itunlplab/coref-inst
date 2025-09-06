"""Configuration utilities for loading and managing settings."""

import os
import yaml
import argparse
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration from YAML files and command line arguments."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        if config_path is None:
            # Default to config/config.yaml relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_yaml_config()
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config if config is not None else {}
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using empty config.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            return {}
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'model.max_seq_length')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def update_from_args(self, args: argparse.Namespace, mapping: Optional[Dict[str, str]] = None) -> None:
        """
        Update configuration with command line arguments.
        
        Args:
            args: Parsed command line arguments
            mapping: Optional mapping from arg names to config paths
        """
        if mapping is None:
            mapping = self._get_default_arg_mapping()
        
        for arg_name, config_path in mapping.items():
            if hasattr(args, arg_name):
                arg_value = getattr(args, arg_name)
                if arg_value is not None:
                    self._set_config_value(config_path, arg_value)
    
    def _get_default_arg_mapping(self) -> Dict[str, str]:
        """Get default mapping from argument names to config paths."""
        return {
            'model_name': 'model.name',
            'max_seq_length': 'model.max_seq_length',
            'load_in_4bit': 'model.load_in_4bit',
            'num_train_epochs': 'training.num_train_epochs',
            'per_device_train_batch_size': 'training.per_device_train_batch_size',
            'learning_rate': 'training.learning_rate',
            'warmup_steps': 'training.warmup_steps',
            'logging_steps': 'training.logging_steps',
            'save_steps': 'training.save_steps',
            'output_dir': 'training.output_dir',
            'dataset_name': 'data.dataset_name',
            'ins_num': 'instructions.instruction_number',
            'seed': 'reproducibility.seed',
            'window_size': 'data.window_size',
            'stride': 'data.stride',
        }
    
    def _set_config_value(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config_section = self.config
        
        # Navigate to the parent section
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the final value
        config_section[keys[-1]] = value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.get('data', {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration section."""
        return self.get('inference', {})
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to YAML file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to config."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return self.get(key) is not None


def setup_reproducibility(config: ConfigLoader, seed: int) -> None:
    """Setup reproducibility settings based on configuration."""
    import random
    import numpy as np
    import torch
    
    # Set seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CUDA settings if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    
    # Reproducibility settings from config
    if config.get('reproducibility.cuda_launch_blocking', True):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    if config.get('reproducibility.cublas_workspace_config'):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = config.get('reproducibility.cublas_workspace_config')
    
    # PyTorch deterministic settings
    if config.get('reproducibility.deterministic', True):
        torch.backends.cudnn.deterministic = True
    
    if not config.get('reproducibility.benchmark', True):
        torch.backends.cudnn.benchmark = False
    
    if not config.get('reproducibility.enabled', True):
        torch.backends.cudnn.enabled = False
    
    if config.get('reproducibility.use_deterministic_algorithms', True):
        torch.use_deterministic_algorithms(True)


def setup_wandb_from_config(config: ConfigLoader, run_name: str, group: str = None) -> None:
    """Setup Weights & Biases from configuration."""
    import wandb
    
    wandb_config = config.get('wandb', {})
    
    # Set environment variables
    os.environ["WANDB__SERVICE_WAIT"] = str(wandb_config.get('service_wait', 300))
    os.environ["WANDB_PROJECT"] = wandb_config.get('project', 'coref-resolution')
    os.environ["WANDB_LOG_MODEL"] = wandb_config.get('log_model', 'checkpoint')
    os.environ["WANDB_WATCH"] = str(wandb_config.get('watch', True)).lower()
    
    # Initialize wandb
    wandb.init(
        name=run_name,
        group=group,
        config=config.config,
        project=wandb_config.get('project', 'coref-resolution')
    )
    
    # Save files if specified
    if wandb_config.get('save_code', True):
        save_files = wandb_config.get('save_files', ['*.py'])
        for pattern in save_files:
            try:
                wandb.save(pattern)
            except Exception as e:
                print(f"Warning: Could not save {pattern} to wandb: {e}")


# Global configuration instance
_global_config = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None or config_path is not None:
        _global_config = ConfigLoader(config_path)
    return _global_config


def reload_config(config_path: Optional[str] = None) -> ConfigLoader:
    """Reload configuration from file."""
    global _global_config
    _global_config = ConfigLoader(config_path)
    return _global_config