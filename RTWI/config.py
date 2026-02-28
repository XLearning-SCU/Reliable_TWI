"""
Configuration for DeepThinkLLM models and sampling parameters.
"""
import re


MODEL_TYPE_CONFIG = {
    "thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "gamma": 0.1, "reasoning_effort": None},
    "instruct": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "gamma": 1.0, "reasoning_effort": "high"},
}


def _extract_model_size(model_name: str) -> int:
    """Extract model size in billions from model name (e.g., '8B', '32B')."""
    match = re.search(r'(\d+)[Bb](?:-|_)', model_name)
    if match:
        return int(match.group(1))
    return 0


def get_sampling_params_from_config(model_name: str) -> dict:
    """
    Get sampling parameters for a model based on type and size.
    
    Args:
        model_name: Model identifier (e.g., 'Qwen3-VL-8B-Instruct', 'Qwen3-VL-32B-Thinking')
        
    Returns:
        Dictionary of sampling parameters
    """
    # Determine model type
    model_type = "thinking" if "thinking" in model_name.lower() else "instruct"
    config = MODEL_TYPE_CONFIG[model_type].copy()
    
    # Adjust temperature for large models (>=32B)
    model_size = _extract_model_size(model_name)
    if model_size >= 32:
        config["temperature"] = 1.0
    
    return config

