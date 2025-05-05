"""
OpenAI Models Explorer - Find and document available OpenAI models

This script does 3 things:
1. Fetches all models your API key has access to
2. Tests a variety of model names to confirm availability 
3. Creates a comprehensive report of capabilities and naming patterns
"""

import os
import json
import time
import requests
from datetime import datetime
from collections import defaultdict

import openai
from openai import OpenAI

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

client = OpenAI(api_key=API_KEY)

# List of model name patterns to try
MODEL_PATTERNS = [
    # GPT-4 Series
    "gpt-4", "gpt-4-32k", "gpt-4-turbo", "gpt-4-vision", "gpt-4o", "gpt-4o-mini",
    # GPT-4 specific versions
    "gpt-4-0314", "gpt-4-0613", "gpt-4-1106-preview", "gpt-4-0125-preview", 
    "gpt-4-turbo-preview", "gpt-4-turbo-2024-04-09",
    # GPT-3.5 Series  
    "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct",
    # Dated GPT-3.5 versions
    "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125",
    # Embedding models
    "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large",
    # Vision models
    "dall-e-2", "dall-e-3",
    # Audio models
    "whisper-1", "tts-1", "tts-1-hd",
    # New nickname convention
    "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-preview",
    # Full prefixes
    "chatgpt-4o", "chatgpt-4o-mini", "chatgpt-4o-latest"
]

# Minimal test prompts
TEST_PROMPTS = {
    "chat": [{"role": "user", "content": "Say hello in one sentence."}],
    "embeddings": "Test sentence for embeddings.",
    "image_gen": "A small robot reading a book, minimal style",
    "audio_transcription": "Test audio transcription",
    "text_to_speech": "This is a test of text to speech."
}

# ────────────────────────────────────────────────────────────────────────────────
# Gather data about available models
# ────────────────────────────────────────────────────────────────────────────────

def get_all_available_models():
    """Fetch all models available to this API key"""
    print("📋 Fetching list of all available models...")
    try:
        models = client.models.list().data
        return sorted([m.id for m in models])
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def categorize_models(models):
    """Group models by their apparent type based on name"""
    categories = defaultdict(list)
    
    for name in models:
        if any(name.startswith(p) for p in ("gpt-4", "gpt-3", "chatgpt-", "o1", "o3")):
            categories["chat"].append(name)
        elif "embedding" in name or name.startswith("text-embedding"):
            categories["embedding"].append(name)
        elif name.startswith(("dall-e", "dall-", "image-")):
            categories["image"].append(name)
        elif name.startswith("tts-") or "-tts" in name:
            categories["text_to_speech"].append(name)
        elif name == "whisper-1" or name.endswith("-transcription"):
            categories["audio_transcription"].append(name)
        else:
            categories["other"].append(name)
            
    return categories

def test_model_access(model_name):
    """Test if a specific model can be accessed with the current API key"""
    
    # Determine what type of model this is based on name patterns
    if any(model_name.startswith(p) for p in ("gpt-", "o1", "o3", "chatgpt-")):
        model_type = "chat"
    elif "embedding" in model_name:
        model_type = "embedding"
    elif any(img in model_name for img in ("dall-e", "dall-", "image-")):
        model_type = "image_gen"
    elif "tts" in model_name:
        model_type = "tts"
    elif "whisper" in model_name or "transcription" in model_name:
        model_type = "transcription"
    else:
        # If we can't determine, try as chat model
        model_type = "chat"
    
    try:
        if model_type == "chat":
            start = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=TEST_PROMPTS["chat"],
                max_tokens=20
            )
            duration = time.time() - start
            return True, duration, None
            
        elif model_type == "embedding":
            start = time.time()
            response = client.embeddings.create(
                model=model_name,
                input=TEST_PROMPTS["embeddings"]
            )
            duration = time.time() - start
            return True, duration, None
            
        # Note: Other model types omitted for brevity
        else:
            # Skip actual API call for other model types to avoid charges
            return False, 0, "Skipped testing for non-chat/embedding model type"
            
    except Exception as e:
        # Log the error details for troubleshooting
        error_info = str(e)
        return False, 0, error_info

# ────────────────────────────────────────────────────────────────────────────────
# Main Execution
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # Get list of all available models
    all_models = get_all_available_models()
    print(f"Found {len(all_models)} models available to your API key")
    
    # Categorize the models
    categories = categorize_models(all_models)
    
    # Initialize results storage
    results = {
        "timestamp": datetime.now().isoformat(),
        "available_models": all_models,
        "categorized_models": {k: v for k, v in categories.items()},
        "model_tests": []
    }
    
    # Test each model pattern
    print("\n🧪 Testing model availability patterns:")
    successful_models = []
    
    for model_name in MODEL_PATTERNS:
        print(f"  Testing '{model_name}'...", end="", flush=True)
        
        success, duration, error = test_model_access(model_name)
        
        if success:
            print(f" ✅ Available ({duration:.2f}s)")
            successful_models.append(model_name)
            results["model_tests"].append({
                "model": model_name,
                "available": True,
                "response_time": duration,
                "error": None
            })
        else:
            if "This model does not exist" in str(error):
                print(f" ❌ Model doesn't exist")
            elif "insufficient_quota" in str(error):
                print(f" ⚠️ Available but requires higher tier plan")
            else:
                print(f" ❌ Error: {error}")
            
            results["model_tests"].append({
                "model": model_name,
                "available": False,
                "response_time": None,
                "error": str(error)
            })
    
    # Summarize findings
    print("\n📊 Summary:")
    print(f"  Total models available to your API key: {len(all_models)}")
    print(f"  Successfully tested models: {len(successful_models)}")
    print(f"  Models by category:")
    for category, models in categories.items():
        print(f"    {category}: {len(models)} models")
    
    # Save results
    filename = f"openai_models_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Report saved to {filename}")
    print("\n✨ Recommended models to use:")
    
    # Get best model in each category
    if categories["chat"]:
        recommended_chat = "gpt-4o" if "gpt-4o" in successful_models else \
                           "gpt-4-turbo" if "gpt-4-turbo" in successful_models else \
                           "gpt-4" if "gpt-4" in successful_models else \
                           "gpt-3.5-turbo"
        print(f"  Chat: {recommended_chat}")
    
    if categories["embedding"]:
        recommended_embedding = "text-embedding-3-large" if "text-embedding-3-large" in successful_models else \
                                "text-embedding-3-small" if "text-embedding-3-small" in successful_models else \
                                "text-embedding-ada-002"
        print(f"  Embeddings: {recommended_embedding}")
        
    # Print common nickname mappings
    print("\n🔤 Model nickname mappings:")
    for nickname, full_name in [
            ("o1", "gpt-4"), 
            ("o3", any(m for m in successful_models if "4o" in m))]:
        if isinstance(full_name, bool):
            # For the o3 case where we check if any 4o model exists
            if full_name:
                # Find the actual name 
                o3_mapping = next((m for m in successful_models if "4o" in m), "unavailable")
                print(f"  {nickname} → {o3_mapping}")
            else:
                print(f"  {nickname} → unavailable")
        else:
            print(f"  {nickname} → {full_name}" + (" (available)" if full_name in successful_models else " (unavailable)"))

if __name__ == "__main__":
    main()