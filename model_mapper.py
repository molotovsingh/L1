"""Helper module to handle model name mapping and parameters for OpenAI models

Model Families:
1. GPT-4 Series: gpt-4, gpt-4-turbo, gpt-4o (multimodal flagship models)
2. o-Series: o1, o3, o4-mini (reasoning-focused models for coding, math, logic)
3. GPT-3.5 Series: gpt-3.5-turbo (legacy mid-tier)
4. Alias endpoints: chatgpt-4o-latest (convenience ID tracking latest build)
"""

def map_model_name(model_name):
    """
    Map model names to their canonical forms if needed
    
    Note: o1 and o3 are their own model families, NOT shortcuts for GPT-4/4o.
    They require special parameters like max_completion_tokens.
    """
    # No mapping needed for o-series models - they are their own model family
    # Only map alias endpoints or handle special cases
    if model_name == "chatgpt-4o-latest":
        return "chatgpt-4o-latest"  # Keep as is
    else:
        return model_name  # Keep all other model names as is

def requires_special_params(model_name):
    """
    Check if a model requires special parameters like max_completion_tokens
    
    The o-series models (o1, o3) require max_completion_tokens instead of max_tokens
    """
    o_series_models = ["o1", "o3", "o4-mini", "o1-mini", "o3-mini", "o1-preview", "o1-pro"]
    return model_name in o_series_models

def get_model_params(model_name, base_params=None):
    """
    Get the correct parameters for a specific model
    
    Args:
        model_name: The model name/ID to use
        base_params: Base parameters dictionary to adjust
        
    Returns:
        Adjusted parameters dictionary suitable for the specified model
    """
    if base_params is None:
        base_params = {}
    
    params = base_params.copy()
    
    # For o-series models, use max_completion_tokens instead of max_tokens
    if requires_special_params(model_name):
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
    
    return params