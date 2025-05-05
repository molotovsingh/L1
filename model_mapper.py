"""Helper module to handle model name mapping"""

def map_model_name(model_name):
    """Map shorthand model names to their full names"""
    if model_name == "o3":
        return "gpt-4o"  # Map o3 to gpt-4o
    elif model_name == "o1":
        return "gpt-4"   # Map o1 to gpt-4
    else:
        return model_name