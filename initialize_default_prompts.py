#!/usr/bin/env python3
"""
Initialize Default Prompts

This script populates the database with default prompt templates from taxonomy_prompts.json.
Run this script after creating the database to ensure default prompts are available.
"""

import json
import os
from db_models import create_custom_prompt, get_custom_prompts

# Default prompts from JSON file
PROMPTS_FILE = "taxonomy_prompts.json"

# Provider-specific versions of prompts
PROVIDERS = ["OpenAI", "Perplexity"]

def read_default_prompts():
    """Read default prompts from the taxonomy_prompts.json file."""
    try:
        with open(PROMPTS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {PROMPTS_FILE}")
        return None
    except json.JSONDecodeError:
        print(f"Error: {PROMPTS_FILE} contains invalid JSON")
        return None

def initialize_default_prompts():
    """Initialize default prompts in the database."""
    # Get existing prompts to check if we already have default prompts
    existing_prompts = get_custom_prompts()
    
    # Check if we have any system prompts 
    has_system_prompts = any(prompt.get("is_system") for prompt in existing_prompts)
    
    if has_system_prompts:
        print("Default prompts already exist in the database. Skipping initialization.")
        return
    
    # Read default prompts from file
    prompts_data = read_default_prompts()
    if not prompts_data:
        return
    
    # Create default Tier-A prompts (candidate generation)
    tier_a_template = prompts_data.get("tier_a_prompt", {}).get("prompt_template", "")
    
    # Create default Tier-B prompts (refinement)
    tier_b_template = prompts_data.get("tier_b_prompt", {}).get("prompt_template", "")
    
    # Create provider-specific versions
    for provider in PROVIDERS:
        # Create Tier-A prompt for this provider
        if tier_a_template:
            create_custom_prompt(
                name=f"Default {provider} Tier-A Prompt",
                tier="A",
                api_provider=provider,
                content=tier_a_template,
                description=f"Default {provider} prompt for Tier-A (candidate generation)",
                is_system=True
            )
            print(f"Created default Tier-A prompt for {provider}")
        
        # Create Tier-B prompt for this provider
        if tier_b_template:
            create_custom_prompt(
                name=f"Default {provider} Tier-B Prompt",
                tier="B",
                api_provider=provider,
                content=tier_b_template,
                description=f"Default {provider} prompt for Tier-B (refinement)",
                is_system=True
            )
            print(f"Created default Tier-B prompt for {provider}")

def main():
    """Main function."""
    print("Initializing default prompts...")
    initialize_default_prompts()
    print("Done.")

if __name__ == "__main__":
    main()