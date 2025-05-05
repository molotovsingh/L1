"""
Simple OpenAI Models Checker - Find the main currently available models
"""

import os
from openai import OpenAI

# Get API key from environment
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# Main model families to check
MAIN_MODELS = [
    # GPT-4 Series: multimodal flagship models
    "gpt-4", 
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    
    # o-Series: reasoning-focused models (separate model family)
    "o1",  # Reasoning-optimized model (NOT a nickname for gpt-4)
    "o3",  # Newer reasoning model, succeeding o1 (NOT a nickname for gpt-4o)
    "o4-mini",  # Mini version in the o-series
    
    # GPT-3.5 Series
    "gpt-3.5-turbo",
    
    # Alias endpoints
    "chatgpt-4o-latest"  # Convenience ID that tracks the latest 4o build
]

def check_model(model_name):
    """Try a simple request to check if the model is available"""
    try:
        # For o-series models, use max_completion_tokens instead of max_tokens
        o_series_models = ["o1", "o3", "o4-mini", "o1-mini", "o3-mini", "o1-preview", "o1-pro"]
        
        if model_name in o_series_models:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_completion_tokens=5
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print("\nCurrent OpenAI Models Availability Check")
    print("---------------------------------------")
    
    # Get all available models
    models = client.models.list().data
    available_models = [m.id for m in models]
    
    print(f"\nFound {len(available_models)} models available to your API key.\n")
    print("Testing key model families:\n")
    
    # Check each model in our main list
    results = []
    for model in MAIN_MODELS:
        success, error = check_model(model)
        status = "✅ Available" if success else "❌ Unavailable"
        if not success and error:
            if "does not exist" in error:
                reason = "Model does not exist"
            elif "deprecated" in error:
                reason = "Model deprecated"
            elif "not a chat model" in error:
                reason = "Not a chat model"
            else:
                reason = error
            status += f" ({reason})"
        
        print(f"{model:20} {status}")
        results.append((model, success))
    
    # Print recommended models
    print("\nRecommended models to use:")
    for family, models, description in [
        ("GPT-4 Series", ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"], 
         "Best for general use and multimodal capabilities"),
        ("o-Series", ["o3", "o1", "o4-mini", "o1-mini", "o3-mini"], 
         "Specialized for reasoning, coding, math and logic tasks"),
        ("GPT-3.5 Series", ["gpt-3.5-turbo"], 
         "Cost-effective for simpler tasks")
    ]:
        available = [m for m in models if m in [r[0] for r in results if r[1]]]
        if available:
            print(f"  {family}: {available[0]} (best), {', '.join(available[1:])} - {description}")

if __name__ == "__main__":
    main()