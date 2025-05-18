"""Configuration loading and processing utilities."""

import os
import importlib.util
from ml_collections import config_dict
from typing import Optional, Union, Dict, Any

from pupperv3_rl.config.defaults import get_default_config

def load_config_from_file(file_path: str) -> config_dict.ConfigDict:
    """Loads configuration from a Python file.
    
    Args:
        file_path: Path to the Python configuration file.
        
    Returns:
        ConfigDict containing the configuration.
    """
    spec = importlib.util.spec_from_file_location("config", file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if hasattr(config_module, "get_config"):
        return config_module.get_config()
    else:
        raise ValueError(f"Configuration file {file_path} does not contain a get_config() function.")

def merge_configs(
    base_config: config_dict.ConfigDict,
    override_config: Optional[config_dict.ConfigDict] = None,
    override_dict: Optional[Dict[str, Any]] = None
) -> config_dict.ConfigDict:
    """Merges configurations with overrides.
    
    Args:
        base_config: Base configuration.
        override_config: Configuration to override the base.
        override_dict: Dictionary of key-value pairs to override.
        
    Returns:
        Merged configuration.
    """
    config = base_config.copy_and_resolve_references()
    
    if override_config is not None:
        # Recursively merge ConfigDict objects
        for key, value in override_config.items():
            if isinstance(value, config_dict.ConfigDict) and key in config:
                config[key] = merge_configs(config[key], value)
            else:
                config[key] = value
    
    if override_dict is not None:
        # Apply flat dictionary of overrides
        for key_path, value in override_dict.items():
            keys = key_path.split(".")
            curr_dict = config
            for key in keys[:-1]:
                if key not in curr_dict:
                    curr_dict[key] = config_dict.ConfigDict()
                curr_dict = curr_dict[key]
            curr_dict[keys[-1]] = value
    
    return config

def load_config(
    config_file: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> config_dict.ConfigDict:
    """Loads and processes configuration.
    
    Args:
        config_file: Path to configuration file.
        overrides: Dictionary of key-value pairs to override.
        
    Returns:
        Processed configuration.
    """
    config = get_default_config()
    
    if config_file is not None and os.path.exists(config_file):
        file_config = load_config_from_file(config_file)
        config = merge_configs(config, file_config)
    
    if overrides is not None:
        config = merge_configs(config, override_dict=overrides)
    
    return config
