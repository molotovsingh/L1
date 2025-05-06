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
import traceback
from typing import Optional, Dict, Any, List

import streamlit as st
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_api_error(error: Exception, model_name: str, api_key: Optional[str], is_tier_a: bool = True):
    """
    Enhanced error logging for Perplexity API calls
    
    Args:
        error: The exception that was raised
        model_name: The model used in the API call
        api_key: The API key used (to check format)
        is_tier_a: Whether this was a Tier-A or Tier-B call
    """
    tier = "Tier-A" if is_tier_a else "Tier-B"
    error_str = str(error)
    
    # Log to console for debugging
    logger.error(f"Perplexity API Error ({tier}): {error_str}")
    
    # Display UI error messages for specific error types
    if "400" in error_str:
        st.warning(f"Bad Request Error in {tier}: The API request format may be incorrect.")
        st.info(f"""
### Debug Info for {tier}:
- **Model:** {model_name}
- **API Base URL:** https://api.perplexity.ai
- **API Key Format:** {'Valid' if api_key and api_key.startswith('pplx-') else 'Invalid or Missing'} 
- **Error Details:** {error_str}

Check that you're using a valid Perplexity model name. Common models include:
- sonar
- sonar-pro
- sonar-reasoning
- sonar-reasoning-pro
- sonar-deep-research
- r1-1776
        """)
        
        # Print the traceback for more details
        st.expander("View Error Traceback", expanded=False).code(traceback.format_exc())

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
    
    # Note: Perplexity model names are used as-is without adding -online suffix
    # The online search capability is built into models like sonar and sonar-pro
    
    st.info(f"🔹 Calling Tier-A (Perplexity) model ({model_name})...")
    
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
        
        # Log model and API call info before making the request
        logger.info(f"Calling Perplexity API (Tier-A) with model: {model_name}")
        
        # Make API call with exact parameter format matching Perplexity documentation
        response = client.chat.completions.create(
            model=model_name,
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
        # Use our enhanced error logging function
        log_api_error(e, model_name, api_key, is_tier_a=True)
        
        # Basic error messaging
        st.error(f"Error calling Perplexity API (Tier-A): {str(e)}")
        
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
    
    # Note: Perplexity model names are used as-is without adding -online suffix
    # The online search capability is built into models like sonar and sonar-pro
    
    st.info(f"🔹 Calling Tier-B (Perplexity) model ({model_name})...")
    
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
        
        # Log model and API call info before making the request
        logger.info(f"Calling Perplexity API (Tier-B) with model: {model_name}")
        
        # Make API call with exact parameter format matching Perplexity documentation
        response = client.chat.completions.create(
            model=model_name,
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
        # Use our enhanced error logging function
        log_api_error(e, model_name, api_key, is_tier_a=False)
        
        # Basic error messaging
        st.error(f"Error calling Perplexity API (Tier-B): {str(e)}")
        
        # Provide helpful error messages based on error type
        if "401" in str(e):
            st.warning("Authentication failed. Check your Perplexity API key.")
        elif "403" in str(e):
            st.warning("Permission denied. Your API key may not have access to this model.")
        elif "429" in str(e):
            st.warning("Rate limit exceeded. Try again later.")
        
        return None