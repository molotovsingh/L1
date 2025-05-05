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
    # GPT-4 family
    "gpt-4", 
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    # GPT-3.5 family
    "gpt-3.5-turbo",
    # Alternative names
    "o1",  # Nickname for gpt-4
    "o3",  # Nickname for gpt-4o
    "chatgpt-4o-latest"
]

def check_model(model_name):
    """Try a simple request to check if the model is available"""
    try:
        # For o1 and o3 shorthand models, use max_completion_tokens instead
        if model_name in ["o1", "o3"]:
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
    for family, models in [
        ("GPT-4 Series", ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"]),
        ("GPT-3.5 Series", ["gpt-3.5-turbo"])
    ]:
        available = [m for m in models if m in [r[0] for r in results if r[1]]]
        if available:
            print(f"  {family}: {available[0]} (best), {', '.join(available[1:])}")

if __name__ == "__main__":
    main()