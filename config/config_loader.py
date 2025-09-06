"""
Configuration loader for Project PokerMind.

This module provides centralized configuration loading from environment variables,
with fallback to configuration files and sensible defaults.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional

# Try to load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not available, use environment variables as-is
    pass


def load_llm_config(config_path: str = "config/llm_config.json") -> Dict[str, Any]:
    """
    Load LLM configuration from environment variables and config file.
    
    Environment variables take precedence over file configuration.
    
    Args:
        config_path: Path to LLM config file
        
    Returns:
        Dictionary containing LLM configuration
    """
    # Default configuration
    config = {
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-needed-for-local",
        "model": "Meta-Llama-3.1-8B-Instruct-Q5_K_M",
        "model_name": "Meta-Llama-3.1-8B-Instruct-Q5_K_M",
        "timeout": 30,
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": False
    }
    
    # Load from file if it exists
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                if "llm_config" in file_config:
                    config.update(file_config["llm_config"])
    except Exception as e:
        print(f"Warning: Failed to load LLM config from {config_path}: {e}")
    
    # Override with environment variables
    config.update({
        "base_url": os.getenv("LLM_BASE_URL", config["base_url"]),
        "api_key": os.getenv("LLM_API_KEY", config["api_key"]),
        "model": os.getenv("LLM_MODEL_NAME", config["model"]),
        "model_name": os.getenv("LLM_MODEL_NAME", config["model_name"]),
        "timeout": int(os.getenv("LLM_TIMEOUT", str(config["timeout"]))),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", str(config["max_tokens"]))),
        "temperature": float(os.getenv("LLM_TEMPERATURE", str(config["temperature"]))),
    })
    
    return config


def load_model_paths() -> Dict[str, str]:
    """
    Load model file paths from environment variables.
    
    Returns:
        Dictionary containing model paths
    """
    return {
        "gto_core": os.getenv("GTO_CORE_MODEL_PATH", "models/gto_core_v1.onnx"),
        "hand_strength": os.getenv("HAND_STRENGTH_MODEL_PATH", "models/hand_strength_estimator.onnx"),
        "gto_preflop": os.getenv("GTO_PREFLOP_MODEL_PATH", "models/gto_preflop_v1.onnx"),
        "gto_river": os.getenv("GTO_RIVER_MODEL_PATH", "models/gto_river_v1.onnx"),
    }


def load_performance_config() -> Dict[str, Any]:
    """
    Load performance configuration from environment variables.
    
    Returns:
        Dictionary containing performance settings
    """
    return {
        "max_inference_time": float(os.getenv("MAX_INFERENCE_TIME", "0.8")),
        "parallel_processing": os.getenv("PARALLEL_PROCESSING", "true").lower() == "true",
        "cuda_available": os.getenv("CUDA_AVAILABLE", "true").lower() == "true",
        "onnx_providers": os.getenv("ONNX_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider").split(","),
    }


def load_logging_config() -> Dict[str, Any]:
    """
    Load logging configuration from environment variables.
    
    Returns:
        Dictionary containing logging settings
    """
    return {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "save_decisions": os.getenv("SAVE_DECISIONS", "true").lower() == "true",
        "save_hand_history": os.getenv("SAVE_HAND_HISTORY", "true").lower() == "true",
        "narration_output_dir": os.getenv("NARRATION_OUTPUT_DIR", "data/hand_history/narrations/"),
    }


def load_training_config() -> Dict[str, Any]:
    """
    Load training configuration from environment variables.
    
    Returns:
        Dictionary containing training settings
    """
    return {
        "training_data_dir": os.getenv("TRAINING_DATA_DIR", "data/training/"),
        "tournament_results_dir": os.getenv("TOURNAMENT_RESULTS_DIR", "tournament_results/"),
    }


def update_agent_config_with_env(config_path: str = "config/agent_config.yaml") -> Dict[str, Any]:
    """
    Load agent configuration and update with environment variable values.
    
    Args:
        config_path: Path to agent config file
        
    Returns:
        Dictionary containing updated agent configuration
    """
    # Load existing config
    config = {}
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load agent config from {config_path}: {e}")
    
    # Update model paths
    model_paths = load_model_paths()
    if "models" not in config:
        config["models"] = {}
    config["models"]["gto_core_path"] = model_paths["gto_core"]
    config["models"]["hand_strength_path"] = model_paths["hand_strength"]
    
    # Update performance settings
    perf_config = load_performance_config()
    if "performance" not in config:
        config["performance"] = {}
    config["performance"]["max_inference_time"] = perf_config["max_inference_time"]
    config["performance"]["parallel_processing"] = perf_config["parallel_processing"]
    config["performance"]["onnx_providers"] = perf_config["onnx_providers"]
    
    # Update logging settings
    log_config = load_logging_config()
    if "logging" not in config:
        config["logging"] = {}
    config["logging"]["level"] = log_config["level"]
    config["logging"]["save_decisions"] = log_config["save_decisions"]
    config["logging"]["save_hand_history"] = log_config["save_hand_history"]
    
    return config


def get_env_or_default(key: str, default: Any, value_type: type = str) -> Any:
    """
    Get environment variable with type conversion and default fallback.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        value_type: Type to convert to (str, int, float, bool)
        
    Returns:
        Environment variable value converted to specified type, or default
    """
    value = os.getenv(key)
    if value is None:
        return default
    
    try:
        if value_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif value_type == int:
            return int(value)
        elif value_type == float:
            return float(value)
        else:
            return str(value)
    except (ValueError, TypeError):
        return default