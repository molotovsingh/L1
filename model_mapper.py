"""Helper module to handle model name mapping"""

def map_model_name(model_name):
    """Map shorthand model names to their full names"""
    if model_name == "o3":
        return "chatgpt-4o-latest"  # Map o3 to chatgpt-4o-latest (which is available)
    elif model_name == "o1":
        return "gpt-4"   # Map o1 to gpt-4
    elif model_name == "gpt-4o":
        return "chatgpt-4o-latest"  # Map gpt-4o to chatgpt-4o-latest
    else:
        return model_name