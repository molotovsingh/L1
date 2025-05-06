"""
Perplexity API integration for taxonomy generation

This module provides functions to call the Perplexity AI API as an alternative
to OpenAI for generating taxonomies. Perplexity offers more reliable results
with reasoning capabilities.
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, List

import streamlit as st
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_perplexity_api_key() -> Optional[str]:
    """Get Perplexity API key from environment."""
    return os.environ.get("PERPLEXITY_API_KEY")

def create_taxonomy_prompt(domain: str, max_labels: int, min_labels: int, deny_list: set) -> str:
    """
    Create a prompt for Perplexity API to generate a taxonomy.
    
    Args:
        domain: The domain for taxonomy generation
        max_labels: Maximum number of labels
        min_labels: Minimum number of labels
        deny_list: Set of denied terms
        
    Returns:
        str: Formatted prompt
    """
    denied_terms = ", ".join(deny_list) if deny_list else "none"
    
    prompt = f"""
You are a domain taxonomy generator specializing in discrete events.

Domain: {domain}

TASK:
Generate a list of {min_labels}–{max_labels} distinct, top-level (L1) categories representing *specific types of events* or *discrete occurrences* within the '{domain}' domain. Think incidents, launches, breakthroughs, failures, breaches, discoveries, major releases, regulatory actions, etc.

Rules for Labels:
1. Format: TitleCase, 1–4 words. May include one internal hyphen (e.g., Model-Launch, Data-Breach, Regulatory-Approval). Start with a capital letter. NO hash symbols (#).
2. Event-Driven: Must describe *what happened* (an event, a change), not an ongoing state, capability, technology area, or general theme.
3. Specificity: Prefer specific event types over overly broad categories.
4. Exclusion: Avoid generic business terms like {denied_terms}. These are handled separately.
5. Output Format: Return ONLY a JSON array of strings. Example: ["Model-Launch", "System-Outage", "Major-Discovery"]

Generate the JSON array now.
"""
    return prompt

def create_taxonomy_audit_prompt(
    domain: str, 
    candidates: List[str], 
    max_labels: int, 
    min_labels: int, 
    deny_list: set
) -> str:
    """
    Create a prompt for Perplexity API to audit and refine a taxonomy.
    
    Args:
        domain: The domain for taxonomy generation
        candidates: List of candidate labels
        max_labels: Maximum number of labels
        min_labels: Minimum number of labels
        deny_list: Set of denied terms
        
    Returns:
        str: Formatted prompt
    """
    denied_terms = ", ".join(deny_list) if deny_list else "none"
    
    prompt = f"""
You are a meticulous taxonomy auditor enforcing specific principles.

Candidate Event Labels for Domain '{domain}':
{json.dumps(candidates, indent=2)}

Your Task:
Review the candidate labels based on the following principles and return a refined list.

Principles to Enforce:
1. Event-Driven Focus: Each label MUST represent a discrete event, incident, change, or occurrence. Reject labels describing general themes, capabilities, technologies, or ongoing states (e.g., "Machine Learning", "Cloud Infrastructure").
2. Formatting: Ensure labels are 1–4 words, TitleCase. Hyphens are allowed ONLY between words (e.g., "Data-Breach" is okay, "AI-Powered" as an event type might be questionable unless it refers to a specific *launch* event). No leading symbols like '#'.
3. Deny List: Reject any label containing the exact terms: {denied_terms}.
4. Consolidation & Target Count: Merge clear synonyms or overly similar event types. Aim for a final list of {max_labels} (±1) distinct, high-value event categories. Prioritize the most significant and common event types for the domain.
5. Output Structure: Return ONLY a JSON object with the following keys:
   - "approved": A JSON array of strings containing the final, approved labels.
   - "rejected": A JSON array of strings containing the labels that were rejected or merged away.
   - "reason_rejected": A JSON object mapping each rejected label (from the "rejected" list) to a brief reason for rejection (e.g., "Not event-driven", "Synonym of X", "Contains denied term", "Too broad").

Example Output Format:
{{
  "approved": ["Model-Launch", "System-Outage", "Regulatory-Action"],
  "rejected": ["AI Research", "Funding Round", "ProductUpdate"],
  "reason_rejected": {{
    "AI Research": "Not event-driven, describes a theme.",
    "Funding Round": "Contains denied term 'Funding'.",
    "ProductUpdate": "Merged into Major-Release."
  }}
}}

Return only the JSON object now.
"""
    return prompt


def call_perplexity_api_tier_a(prompt: str, api_key: Optional[str], model_name: str = "sonar") -> Optional[str]:
    """
    Call Perplexity API for Tier-A candidate generation.
    
    Args:
        prompt: Prompt for the API
        api_key: Perplexity API key
        model_name: Perplexity model to use
        
    Returns:
        str or None: API response content or None on failure
    """
    if not api_key:
        st.error("PERPLEXITY_API_KEY required but not found in environment variables.")
        return None
    
    # Check if key has proper format
    if not api_key.startswith("pplx-"):
        st.warning("Perplexity API key doesn't have the expected format (should start with 'pplx-')")
    
    # Handle online vs offline models
    # Models that don't use web search
    offline_models = ["r1-1776"]
    
    # Models that typically use web search (add -online suffix)
    search_models = ["sonar", "sonar-pro"]
    
    # Use the model name directly from the Perplexity documentation
    model_to_use = model_name
    
    # Add -online suffix only for search models that don't already have it
    if model_name in search_models and not model_name.endswith("-online"):
        model_to_use = f"{model_name}-online"
    
    st.info(f"🔹 Calling Tier-A (Perplexity) model ({model_to_use})...")
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        
        # Create system message and user message
        messages = [
            {
                "role": "system",
                "content": "You are a domain taxonomy generator specializing in structured taxonomies and event categorization."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Make API call
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            top_p=1,
        )
        
        # Extract content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content.strip()
        
        st.error(f"Perplexity API returned an empty or invalid response for Tier-A.")
        return None
        
    except Exception as e:
        st.error(f"Error calling Perplexity API: {e}")
        
        # Provide helpful error messages based on error type
        if "401" in str(e):
            st.warning("Authentication failed. Check your Perplexity API key.")
        elif "403" in str(e):
            st.warning("Permission denied. Your API key may not have access to this model.")
        elif "429" in str(e):
            st.warning("Rate limit exceeded. Try again later.")
        
        return None


def call_perplexity_api_tier_b(prompt: str, api_key: Optional[str], model_name: str = "sonar-reasoning") -> Optional[str]:
    """
    Call Perplexity API for Tier-B refinement.
    
    Args:
        prompt: Prompt for the API
        api_key: Perplexity API key
        model_name: Perplexity model to use
        
    Returns:
        str or None: API response content or None on failure
    """
    if not api_key:
        st.error("PERPLEXITY_API_KEY required but not found in environment variables.")
        return None
    
    # Check if key has proper format
    if not api_key.startswith("pplx-"):
        st.warning("Perplexity API key doesn't have the expected format (should start with 'pplx-')")
    
    # Handle online vs offline models
    # Models that don't use web search
    offline_models = ["r1-1776"]
    
    # Models that typically use web search (add -online suffix)
    search_models = ["sonar", "sonar-pro"]
    
    # Use the model name directly from the Perplexity documentation
    model_to_use = model_name
    
    # Add -online suffix only for search models that don't already have it
    if model_name in search_models and not model_name.endswith("-online"):
        model_to_use = f"{model_name}-online"
    
    st.info(f"🔹 Calling Tier-B (Perplexity) model ({model_to_use})...")
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        
        # Create system message and user message
        messages = [
            {
                "role": "system",
                "content": "You are a meticulous taxonomy auditor specializing in structured taxonomies and event validation."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Make API call
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            top_p=1,
        )
        
        # Extract content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content.strip()
        
        st.error(f"Perplexity API returned an empty or invalid response for Tier-B.")
        return None
        
    except Exception as e:
        st.error(f"Error calling Perplexity API: {e}")
        
        # Provide helpful error messages based on error type
        if "401" in str(e):
            st.warning("Authentication failed. Check your Perplexity API key.")
        elif "403" in str(e):
            st.warning("Permission denied. Your API key may not have access to this model.")
        elif "429" in str(e):
            st.warning("Rate limit exceeded. Try again later.")
        
        return None