"""Helper module to handle model name mapping and parameters for OpenAI models

Model Families:
1. GPT-4 Series (multimodal): gpt-4, gpt-4-turbo, gpt-4o (multimodal flagship models)
   - Strengths: Image+audio input, TTS, web-search preview, faster tokens
   - Complete feature set (vision, audio, search)
   
2. o-Series (reasoning): o1, o3, o4-mini (reasoning-focused models)
   - Strengths: Long-form logic, code, math, structured thinking
   - Limitations: Text-only, no vision/audio, may return empty responses
   - Parameter differences: Uses max_completion_tokens instead of max_tokens
   
3. GPT-3.5 Series: gpt-3.5-turbo (legacy mid-tier)
   - Most reliable for general use and highest rate limits

4. Alias endpoints: chatgpt-4o-latest (convenience ID tracking latest build)

Key differences:
- Both use same chat/completions endpoint but parameters differ
- o-series requires special access permissions for content generation
- o-series models typically outperform GPT-4 on deep reasoning tasks
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
    
    # Handle o-series model requirements
    o_series_models = ["o1", "o3", "o4-mini", "o1-mini", "o3-mini", "o1-preview", "o1-pro"]
    if model_name in o_series_models:
        # 1. Use max_completion_tokens instead of max_tokens
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
            
        # 2. Remove temperature parameter - o-series only support default temp (1.0)
        if "temperature" in params:
            params.pop("temperature")
    
    return params