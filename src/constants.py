"""Constants and configuration for the coreference resolution system."""

from typing import Dict, Any, Optional
from .config_utils import get_config


def get_language_dict() -> Dict[str, str]:
    """Get language mappings from configuration."""
    config = get_config()
    return config.get('languages.language_dict', {
        "tr": "Turkish",
        "ca": "Catalan", 
        "cs": "Czech",
        "cu": "Church Slavonic",
        "de": "German",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "grc": "Ancient Greek",
        "hbo": "Ancient Hebrew",
        "hu": "Hungarian",
        "lt": "Lithuanian",
        "no": "Norwegian",
        "pl": "Polish",
        "ru": "Russian"
    })


def get_no_zero_languages() -> list:
    """Get list of languages without zero mentions."""
    config = get_config()
    return config.get('languages.no_zero_langs', ['en', 'fr', 'ru', 'no', 'lt', 'hbo'])


def get_zero_mention_instruction() -> str:
    """Get zero mention instruction text."""
    config = get_config()
    return config.get('instructions.zero_mention_text', 
        "Where you see </z>@ there is a zero mention, which is normally not written "
        "but you also need to link them with other mentions."
    )


def get_model_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Get model configuration."""
    config = get_config()
    model_config = config.get_model_config()
    
    # If specific model name provided, get its settings
    if model_name and 'supported_models' in model_config:
        supported = model_config['supported_models']
        if model_name in supported:
            model_config['model_path'] = supported[model_name]
    
    return model_config


def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    config = get_config()
    return config.get_training_config()


def get_data_config() -> Dict[str, Any]:
    """Get data processing configuration."""
    config = get_config()
    return config.get_data_config()


def get_inference_config() -> Dict[str, Any]:
    """Get inference configuration."""
    config = get_config()
    return config.get_inference_config()


def get_model_type(model_name: str) -> str:
    """Extract model type from model name."""
    model_name_lower = model_name.lower()
    if 'gemma' in model_name_lower:
        return 'gemma'
    elif 'mistral' in model_name_lower:
        return 'mistral'
    elif 'llama' in model_name_lower:
        return 'llama'
    else:
        raise ValueError(f"Unknown model type for: {model_name}")


# Backward compatibility - these will use config values
LANGUAGE_DICT = get_language_dict()
NO_ZERO_LANG = get_no_zero_languages()
ZERO_MENTION_INSTRUCTION = get_zero_mention_instruction()
MODEL_CONFIGS = get_model_config()
TRAINING_DEFAULTS = get_training_config()