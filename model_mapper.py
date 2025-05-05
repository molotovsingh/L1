"""Helper module to handle model name mapping"""

def map_model_name(model_name):
    """Map shorthand model names to their full names"""
    # New OpenAI behavior: o1 and o3 shorthands work directly
    # but require max_completion_tokens instead of max_tokens
    if model_name == "o3":
        # Both work, but using o3 directly allows for special parameters
        return "o3"  # Keep as o3, don't map to chatgpt-4o-latest
    elif model_name == "o1":
        # Both work, but using o1 directly allows for special parameters
        return "o1"  # Keep as o1, don't map to gpt-4
    else:
        return model_name

def requires_special_params(model_name):
    """Check if a model requires special parameters like max_completion_tokens"""
    return model_name in ["o1", "o3"]

def get_model_params(model_name, base_params=None):
    """Get the correct parameters for a specific model"""
    if base_params is None:
        base_params = {}
    
    params = base_params.copy()
    
    # For o1 and o3 we need to use max_completion_tokens instead of max_tokens
    if requires_special_params(model_name):
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
    
    return params