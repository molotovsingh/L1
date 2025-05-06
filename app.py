# --- Taxonomy Discovery App ---
# A Streamlit application for interactive domain taxonomy generation 
# using OpenAI API for both tiers

import os
import re
import json
import datetime
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

# Third-party Libraries
import streamlit as st

# Database models
import db_models

# Custom utilities
import model_mapper
import call_apis
import call_perplexity_api

# API clients
try:
    from openai import OpenAI, APIError as OpenAI_APIError, AuthenticationError as OpenAI_AuthError
    from openai import RateLimitError as OpenAI_RateLimitError, APIConnectionError as OpenAI_ConnError
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

# Global settings for API retry logic
MAX_RETRIES = 4
RETRY_DELAYS = [5, 15, 30, 60]  # Increased exponential backoff in seconds (up to 1 minute)

# ----- Configuration (Defaults) -----
# API providers
API_PROVIDERS = ["OpenAI", "Perplexity"]

# OpenAI models for both tiers
# The newest OpenAI model is "gpt-4o" which was released May 13, 2024
DEFAULT_OPENAI_TIER_A_OPTIONS: List[str] = [
    "gpt-4o",             # Powerful, reliable model (OpenAI's latest model)
    "gpt-4",              # Reliable model
    "gpt-3.5-turbo",      # Faster, more economical
    "custom",             # Allow user to specify a custom model
    "o3",                 # (NOT RECOMMENDED - may return empty responses)
    "o1"                  # (NOT RECOMMENDED - may return empty responses)
]

DEFAULT_OPENAI_TIER_B_OPTIONS: List[str] = [
    "gpt-4o",             # Powerful, reliable model (OpenAI's latest model)
    "gpt-4",              # Reliable model
    "gpt-3.5-turbo",      # Faster, more economical
    "custom",             # Allow user to specify a custom model
    "None/Offline",       # Skip Tier-B processing
    "o3",                 # (NOT RECOMMENDED - may return empty responses)
    "o1"                  # (NOT RECOMMENDED - may return empty responses)
]

# Perplexity models (as of May 2025)
DEFAULT_PERPLEXITY_TIER_A_OPTIONS: List[str] = [
    "sonar",              # Lightweight, cost-effective search model
    "sonar-pro",          # Advanced search with grounding
    "sonar-deep-research", # Expert-level research model
    "sonar-reasoning",    # Fast, real-time reasoning model
    "sonar-reasoning-pro", # Premier reasoning with Chain of Thought
    "r1-1776",            # Offline model (no search)
    "custom"              # Allow user to specify a custom model
]

DEFAULT_PERPLEXITY_TIER_B_OPTIONS: List[str] = [
    "sonar-reasoning",    # Fast, real-time reasoning (recommended for refinement)
    "sonar-reasoning-pro", # Premier reasoning with Chain of Thought
    "sonar-deep-research", # Expert-level research model
    "sonar",              # Lightweight, cost-effective search model
    "sonar-pro",          # Advanced search with grounding
    "r1-1776",            # Offline model (no search)
    "custom",             # Allow user to specify a custom model
    "None/Offline"        # Skip Tier-B processing
]

# Default to OpenAI options for backward compatibility
DEFAULT_TIER_A_OPTIONS = DEFAULT_OPENAI_TIER_A_OPTIONS
DEFAULT_TIER_B_OPTIONS = DEFAULT_OPENAI_TIER_B_OPTIONS
DEFAULT_MAX_LABELS: int = 9
DEFAULT_MIN_LABELS: int = 8
DEFAULT_DENY_LIST: str = "Funding\nHiring\nPartnership"
DEFAULT_OUT_DIR: str = "taxonomies"

# ----- Setup Logging -----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- Helper Functions for API Calls -----
# Moved to call_apis.py


# ----- Core Taxonomy Generation Logic -----
def generate_taxonomy(domain: str, tier_a_model: str, tier_b_model: str, max_labels: int, min_labels: int, 
                      deny_list: set, out_dir: Path, api_provider: str = "OpenAI", 
                      openai_api_key: Optional[str] = None, perplexity_api_key: Optional[str] = None):
    """
    The main function to generate and validate the taxonomy using APIs.
    
    Args:
        domain: The domain for which to generate a taxonomy
        tier_a_model: The model to use for candidate generation
        tier_b_model: The model to use for refinement
        max_labels: Maximum number of labels in the taxonomy
        min_labels: Minimum number of labels in the taxonomy
        deny_list: Set of terms to exclude from the taxonomy
        out_dir: Directory to save the taxonomy
        api_provider: Which API provider to use ("OpenAI" or "Perplexity")
        openai_api_key: OpenAI API key (if api_provider is "OpenAI")
        perplexity_api_key: Perplexity API key (if api_provider is "Perplexity")
        
    Returns:
        Tuple of approved labels, rejected labels, and rejection reasons
        
    Note on o-series models (o1, o3, etc.):
    These models require special access permissions to return content. In many cases,
    they may return empty responses even though the API call succeeds. If you encounter
    empty responses, consider using standard GPT models like gpt-4o or gpt-3.5-turbo,
    or switch to Perplexity API which provides more reliable results.
    """
    # Check for o-series models and display warning
    o_series_models = ["o1", "o3", "o4-mini", "o1-mini", "o3-mini", "o1-preview", "o1-pro"]
    if tier_a_model in o_series_models or tier_b_model in o_series_models:
        st.warning("""
        ⚠️ **Note about o-series models**: 
        
        The o-series models (o1, o3, etc.) may return empty responses depending on account permissions
        even though the API call succeeds. If you encounter errors, consider using standard models
        like gpt-4o, gpt-4, or gpt-3.5-turbo instead.
        """)
    
    if not domain:
        st.error("Domain input cannot be empty.")
        return None, None, None

    st.info(f"Processing domain: {domain}")
    out_dir.mkdir(exist_ok=True)

    # ----- Tier‑A candidate generation -----
    if api_provider == "OpenAI":
        prompt_A = f"""
You are a domain taxonomy generator specializing in discrete events.

Domain: {domain}

TASK:
Generate a list of 12–15 distinct, top-level (L1) categories representing *specific types of events* or *discrete occurrences* within the '{domain}' domain. Think incidents, launches, breakthroughs, failures, breaches, discoveries, major releases, regulatory actions, etc.

Rules for Labels:
1. Format: TitleCase, 1–4 words. May include one internal hyphen (e.g., Model-Launch, Data-Breach, Regulatory-Approval). Start with a capital letter. NO hash symbols (#).
2. Event-Driven: Must describe *what happened* (an event, a change), not an ongoing state, capability, technology area, or general theme.
3. Specificity: Prefer specific event types over overly broad categories.
4. Exclusion: Avoid generic business terms like {', '.join(deny_list)}. These are handled separately.
5. Output Format: Return ONLY a JSON array of strings. Example: ["Model-Launch", "System-Outage", "Major-Discovery"]

Generate the JSON array now.
"""

        st.info(f"🔹 Generating Tier‑A candidates via {api_provider} API...")
        with st.spinner(f"Waiting for {api_provider} API response for generation..."):
            candidates: List[str] = []
            resp_A = call_apis.call_tier_a_api(prompt_A, openai_api_key, tier_a_model)
            
    else:  # Perplexity
        # Create prompt for Perplexity
        prompt_A = call_perplexity_api.create_taxonomy_prompt(domain, max_labels, min_labels, deny_list)
        
        st.info(f"🔹 Generating Tier‑A candidates via {api_provider} API...")
        with st.spinner(f"Waiting for {api_provider} API response for generation..."):
            candidates: List[str] = []
            resp_A = call_perplexity_api.call_perplexity_api_tier_a(prompt_A, perplexity_api_key, tier_a_model)

    if resp_A:
        # Display raw response in expander
        with st.expander("Raw Tier-A Response"):
            st.code(resp_A, language="json")

        # Attempt to extract JSON - LLMs sometimes add preamble/postamble text
        json_match = re.search(r'\[.*?\]', resp_A, re.DOTALL | re.IGNORECASE)  # Find bracketed list more robustly
        if json_match:
            json_str = json_match.group(0)
            try:
                candidates_raw = json.loads(json_str)
                # Clean candidates robustly
                candidates = [str(c).strip().lstrip('# ') for c in candidates_raw if isinstance(c, str) and str(c).strip()]
                st.success(f"✅ Tier-A proposed {len(candidates)} labels (extracted JSON)")
            except json.JSONDecodeError:
                st.warning(f"Tier-A JSON structure invalid in extracted part. Trying full response.")
                # Fallback to parsing the whole response carefully
                try:
                    # A final attempt - maybe it's just the array without brackets in text
                    if resp_A.strip().startswith('"') and resp_A.strip().endswith('"'):
                        resp_A_maybe_list = f"[{resp_A}]"  # Wrap in brackets if it looks like comma-sep strings
                    else:
                        resp_A_maybe_list = resp_A  # Try as is
                    candidates_raw = json.loads(resp_A_maybe_list)
                    candidates = [str(c).strip().lstrip('# ') for c in candidates_raw if isinstance(c, str) and str(c).strip()]
                    st.success(f"✅ Tier-A proposed {len(candidates)} labels (full response parse)")
                except json.JSONDecodeError:
                    st.error(f"Tier‑A returned unparsable JSON even on fallback.")
                    return None, None, None
        else:
            # Maybe the LLM ignored the JSON request and just gave a list
            lines = [line.strip().lstrip('- ').lstrip('* ').lstrip('# ') for line in resp_A.split('\n') if line.strip()]
            # Basic check if lines look like labels
            if lines and len(lines) > 3 and all(1 <= len(line.split()) <= 5 for line in lines):
                candidates = lines
                st.warning(f"Tier-A did not return JSON, but parsed {len(candidates)} lines as potential labels")
            else:
                st.error(f"Tier-A did not return a recognizable JSON array or list.")
                return None, None, None
    else:
        st.error("Tier-A generation failed (API call error or empty response).")
        return None, None, None  # Stop processing

    if not candidates:
        st.error("Tier-A generation resulted in zero valid candidates.")
        
        # If this is because we used an o-series model, add additional guidance
        o_series_models = ["o1", "o3", "o4-mini", "o1-mini", "o3-mini", "o1-preview", "o1-pro"]
        if tier_a_model in o_series_models:
            st.error("""
            ❌ **O-series Model Not Supported**
            
            The taxonomy generation failed because the o-series model returned an empty response.
            This is a common issue with o-series models in certain accounts.
            
            **Solution:**
            1. Go back and select a standard GPT model like `gpt-4o` or `gpt-3.5-turbo`
            2. Click 'Generate Taxonomy' again
            
            The o-series models require special access permissions to return content, and in many cases,
            they will return empty responses even though the API call succeeds with status 200 OK.
            """)
            
        return None, None, None

    # Display the candidates
    st.subheader("Tier-A Candidates")
    st.write(candidates)

    # ----- Tier‑B audit & refinement -----
    approved: List[str] = []
    rejected: List[str] = []
    rejected_info: Dict[str, str] = {}
    tier_b_selected_model = tier_b_model

    if tier_b_selected_model.lower() != "none/offline":
        if api_provider == "OpenAI":
            prompt_B = f"""
You are a meticulous taxonomy auditor enforcing specific principles.

Candidate Event Labels for Domain '{domain}':
{json.dumps(candidates, indent=2)}

Your Task:
Review the candidate labels based on the following principles and return a refined list.

Principles to Enforce:
1. Event-Driven Focus: Each label MUST represent a discrete event, incident, change, or occurrence. Reject labels describing general themes, capabilities, technologies, or ongoing states (e.g., "Machine Learning", "Cloud Infrastructure").
2. Formatting: Ensure labels are 1–4 words, TitleCase. Hyphens are allowed ONLY between words (e.g., "Data-Breach" is okay, "AI-Powered" as an event type might be questionable unless it refers to a specific *launch* event). No leading symbols like '#'.
3. Deny List: Reject any label containing the exact terms: {', '.join(deny_list)}.
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
            with st.spinner(f"Waiting for {api_provider} API response for refinement..."):
                audit_response_str = call_apis.call_openai_api(prompt_B, openai_api_key, tier_b_selected_model)
                
        else:  # Perplexity
            # Create prompt for Perplexity
            prompt_B = call_perplexity_api.create_taxonomy_audit_prompt(domain, candidates, max_labels, min_labels, deny_list)
            
            with st.spinner(f"Waiting for {api_provider} API response for refinement..."):
                audit_response_str = call_perplexity_api.call_perplexity_api_tier_b(prompt_B, perplexity_api_key, tier_b_selected_model)

        if audit_response_str:
            # Display raw response in expander
            with st.expander("Raw Tier-B Response"):
                st.code(audit_response_str, language="json")

            try:
                # First attempt - direct JSON parsing
                try:
                    audit_result = json.loads(audit_response_str)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON object pattern
                    st.warning("Direct JSON parsing failed. Attempting to extract JSON object from response.")
                    # Look for JSON object pattern in the response {....}
                    json_match = re.search(r'\{.*\}', audit_response_str, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        try:
                            audit_result = json.loads(json_str)
                            st.info("Successfully extracted JSON object from response.")
                        except json.JSONDecodeError:
                            # If still fails, display detailed error
                            st.error("Extracted content is not valid JSON.")
                            st.expander("Extracted Content").code(json_str)
                            raise
                    else:
                        st.error("Could not find JSON object pattern in response.")
                        raise json.JSONDecodeError("No JSON object found", audit_response_str, 0)
                
                # Verify expected structure
                if isinstance(audit_result, dict) and \
                   "approved" in audit_result and isinstance(audit_result["approved"], list) and \
                   "rejected" in audit_result and isinstance(audit_result["rejected"], list) and \
                   "reason_rejected" in audit_result and isinstance(audit_result["reason_rejected"], dict):

                    approved = [str(lbl).strip().lstrip('# ') for lbl in audit_result["approved"] if isinstance(lbl, str)]
                    rejected = [str(lbl).strip().lstrip('# ') for lbl in audit_result["rejected"] if isinstance(lbl, str)]
                    rejected_info = audit_result["reason_rejected"]
                    st.success(f"✅ Tier-B approved {len(approved)} labels after audit.")
                else:
                    st.warning("Tier-B response JSON structure is invalid. Falling back to Tier-A candidates.")
                    st.expander("Invalid JSON Structure").code(str(audit_result))
                    approved = candidates
            except json.JSONDecodeError as e:
                st.error(f"Tier-B returned unparsable JSON: {e}")
                # Display more details about the error
                st.expander("JSON Parsing Error Details").info(f"""
                - Error Message: {str(e)}
                - Error Position: {e.pos}
                - Line Number: {audit_response_str.count(chr(10), 0, e.pos) + 1}
                """)
                approved = candidates  # Fallback to using all candidates
        else:
            st.warning("No Tier-B refinement performed. Using Tier-A candidates as final.")
            approved = candidates
            
            # If this is because we used an o-series model, add additional guidance
            o_series_models = ["o1", "o3", "o4-mini", "o1-mini", "o3-mini", "o1-preview", "o1-pro"]
            if tier_b_model in o_series_models:
                st.warning("""
                ⚠️ **O-series Model Note:** 
                
                The Tier-B refinement with an o-series model failed to return usable content.
                This is a common limitation of these models with some account permissions.
                
                **Recommendations:**
                - Try using standard GPT models like `gpt-4o` or `gpt-3.5-turbo` instead
                - Check the "Model Info" tab for details about model capabilities and limitations
                """)
    else:
        st.info("Tier-B refinement skipped (None/Offline selected). Using Tier-A candidates as final.")
        approved = candidates

    # Save the taxonomy
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{domain.replace(' ', '_')}_{timestamp}.json"
    output_file = out_dir / filename
    
    taxonomy_data = {
        "domain": domain,
        "timestamp": timestamp,
        "api_provider": api_provider,
        "approved_labels": approved,
        "rejected_labels": rejected if rejected else [],
        "rejection_reasons": rejected_info if rejected_info else {},
        "tier_a_model": tier_a_model,
        "tier_b_model": tier_b_model,
        "max_labels": max_labels,
        "min_labels": min_labels,
        "deny_list": list(deny_list)
    }
    
    # Save to file system
    try:
        with open(output_file, "w") as f:
            json.dump(taxonomy_data, f, indent=2)
        st.success(f"Taxonomy saved to file: {output_file}")
    except Exception as e:
        st.error(f"Failed to save taxonomy to file: {e}")
    
    # Save to database with improved error handling
    taxonomy_id = db_models.create_taxonomy(
        domain=domain,
        tier_a_model=tier_a_model,
        tier_b_model=tier_b_model,
        max_labels=max_labels,
        min_labels=min_labels,
        deny_list=deny_list,
        approved_labels=approved,
        rejected_labels=rejected if rejected else [],
        rejection_reasons=rejected_info if rejected_info else {},
        api_provider=api_provider
    )
    
    if taxonomy_id:
        st.success(f"✅ Taxonomy saved to database with ID: {taxonomy_id}")
    else:
        st.warning("⚠️ Taxonomy was saved to file but not to database due to a connection issue.")
        st.info("Your data is safe, but won't appear in the 'View Previous Taxonomies' tab until database connectivity is restored.")

    return approved, rejected, rejected_info


# ----- Streamlit App UI -----
def main():
    st.set_page_config(
        page_title="Taxonomy Discovery App",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Interactive Domain Taxonomy Discovery")
    st.markdown("""
    This app helps you generate taxonomies for any domain using a two-tier approach with multiple API providers:
    - **Tier-A**: Generates candidate labels (OpenAI or Perplexity API)
    - **Tier-B**: Refines and validates the taxonomy (OpenAI or Perplexity API)
    
    Use the tabs below to generate a new taxonomy or view previously generated ones.
    """)
    
    # API Key warnings/status (outside tabs)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
    
    # API key status
    col1_api, col2_api = st.columns(2)
    
    with col1_api:
        if openai_api_key:
            st.success("✅ OPENAI_API_KEY found in environment variables")
        else:
            st.error("❌ OPENAI_API_KEY not found. Set it in your environment variables.")
            
    with col2_api:
        if perplexity_api_key:
            st.success("✅ PERPLEXITY_API_KEY found in environment variables")
        else:
            st.warning("⚠️ PERPLEXITY_API_KEY not found. Only OpenAI API will be available.")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Generate Taxonomy", "View Previous Taxonomies", "Model Info"])
    
    with tab1:
        st.header("Generate New Taxonomy")
        st.markdown("Enter your domain and configuration parameters below to start.")

        # API provider selection outside the form to make model selection dynamic
        api_provider = st.selectbox(
            "API Provider",
            options=API_PROVIDERS,
            index=0,
            help="Select which API provider to use for generating the taxonomy"
        )
        
        # Display warning if Perplexity is selected but no API key is available
        if api_provider == "Perplexity" and not perplexity_api_key:
            st.warning("⚠️ No Perplexity API key found. Please add a PERPLEXITY_API_KEY to your environment variables.")
        
        # Input Form
        with st.form("taxonomy_config_form"):
            domain = st.text_input("Domain", help="Enter the domain for which you want to generate a taxonomy (e.g., 'Artificial Intelligence', 'Healthcare Tech')")
            
            # Hidden field to store API provider selection
            api_provider_hidden = st.text_input("API Provider Hidden", value=api_provider, key="api_provider_hidden", label_visibility="collapsed")
            
            # Advanced settings expander
            with st.expander("Advanced Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Different model options based on the selected provider
                    if api_provider == "OpenAI":
                        tier_a_model_option = st.selectbox(
                            "Tier-A Model (OpenAI)", 
                            options=DEFAULT_OPENAI_TIER_A_OPTIONS,
                            index=0,
                            help="Select the OpenAI model for candidate generation. Note: o-series models (o1, o3) may return empty responses depending on account permissions. GPT-4o or GPT-3.5-turbo are recommended."
                        )
                        
                        # If custom is selected, show an input field for custom model name
                        if tier_a_model_option == "custom":
                            tier_a_custom_model = st.text_input(
                                "Custom Tier-A Model Name",
                                value="gpt-4o-mini",
                                help="Enter a valid OpenAI model name (e.g., gpt-4o-mini, gpt-4-turbo, etc.)"
                            )
                            tier_a_model = tier_a_custom_model
                        else:
                            tier_a_model = tier_a_model_option
                    else:  # Perplexity
                        tier_a_model_option = st.selectbox(
                            "Tier-A Model (Perplexity)", 
                            options=DEFAULT_PERPLEXITY_TIER_A_OPTIONS,
                            index=0,
                            help="Select the Perplexity model for candidate generation. Perplexity models generally provide more reliable results with reasoning capabilities."
                        )
                        
                        # If custom is selected, show an input field for custom model name
                        if tier_a_model_option == "custom":
                            tier_a_custom_model = st.text_input(
                                "Custom Tier-A Model Name",
                                value="sonar-pro",
                                help="Enter a valid Perplexity model name (e.g., sonar-pro, codestral-2305, r1-1776)"
                            )
                            tier_a_model = tier_a_custom_model
                        else:
                            tier_a_model = tier_a_model_option
                    
                    max_labels = st.number_input(
                        "Max Labels", 
                        min_value=5, 
                        max_value=15, 
                        value=DEFAULT_MAX_LABELS,
                        help="Maximum number of labels in the final taxonomy"
                    )
                    
                    deny_list_text = st.text_area(
                        "Deny List (one term per line)", 
                        value=DEFAULT_DENY_LIST,
                        height=100,
                        help="Terms to exclude from the taxonomy (e.g., Funding, Hiring)"
                    )
                
                with col2:
                    # Different model options based on the selected provider
                    if api_provider == "OpenAI":
                        tier_b_model_option = st.selectbox(
                            "Tier-B Model (OpenAI)", 
                            options=DEFAULT_OPENAI_TIER_B_OPTIONS,
                            index=0,
                            help="Select the OpenAI model for refinement (or None/Offline to skip). Note: o-series models (o1, o3) may return empty responses depending on account permissions. GPT-4o or GPT-3.5-turbo are recommended."
                        )
                        
                        # If custom is selected, show an input field for custom model name
                        if tier_b_model_option == "custom":
                            tier_b_custom_model = st.text_input(
                                "Custom Tier-B Model Name",
                                value="gpt-4o-mini",
                                help="Enter a valid OpenAI model name (e.g., gpt-4o-mini, gpt-4-turbo, etc.)"
                            )
                            tier_b_model = tier_b_custom_model
                        else:
                            tier_b_model = tier_b_model_option
                    else:  # Perplexity
                        tier_b_model_option = st.selectbox(
                            "Tier-B Model (Perplexity)", 
                            options=DEFAULT_PERPLEXITY_TIER_B_OPTIONS,
                            index=0,
                            help="Select the Perplexity model for refinement (or None/Offline to skip). Perplexity models generally provide more reliable results with reasoning capabilities."
                        )
                        
                        # If custom is selected, show an input field for custom model name
                        if tier_b_model_option == "custom":
                            tier_b_custom_model = st.text_input(
                                "Custom Tier-B Model Name",
                                value="sonar-reasoning-pro",
                                help="Enter a valid Perplexity model name (e.g., sonar-reasoning-pro, codestral-2305, r1-1776)"
                            )
                            tier_b_model = tier_b_custom_model
                        else:
                            tier_b_model = tier_b_model_option
                    
                    min_labels = st.number_input(
                        "Min Labels", 
                        min_value=3, 
                        max_value=12, 
                        value=DEFAULT_MIN_LABELS,
                        help="Minimum number of labels in the final taxonomy"
                    )
                    
                    out_dir_str = st.text_input(
                        "Output Directory", 
                        value=DEFAULT_OUT_DIR,
                        help="Directory to save the generated taxonomies"
                    )
            
            submit_button = st.form_submit_button("Generate Taxonomy")
        
        # Process form submission
        if submit_button:
            if not domain:
                st.error("Please enter a domain to generate a taxonomy.")
                return
                
            # Process deny list
            deny_list = set(line.strip() for line in deny_list_text.split('\n') if line.strip())
            
            # Create output directory
            out_dir = Path(out_dir_str)
            
            # Status container for updates
            status_container = st.empty()
            
            with status_container.container():
                # Use the api_provider from hidden field (matches what was selected when form was populated)
                form_api_provider = api_provider_hidden
                
                # Generate taxonomy based on selected API provider
                approved, rejected, rejection_reasons = generate_taxonomy(
                    domain=domain,
                    tier_a_model=tier_a_model,
                    tier_b_model=tier_b_model,
                    max_labels=max_labels,
                    min_labels=min_labels,
                    deny_list=deny_list,
                    out_dir=out_dir,
                    api_provider=form_api_provider,
                    openai_api_key=openai_api_key,
                    perplexity_api_key=perplexity_api_key
                )
                
                if approved:
                    # Display final taxonomy
                    st.subheader(f"Final Taxonomy for '{domain}'")
                    
                    # Display approved labels
                    st.success(f"✅ {len(approved)} Approved Labels:")
                    for label in approved:
                        st.write(f"- {label}")
                    
                    # Display rejected labels and reasons
                    if rejected and rejection_reasons:
                        with st.expander(f"⛔ {len(rejected)} Rejected Labels"):
                            for label in rejected:
                                reason = rejection_reasons.get(label, "No reason provided")
                                st.write(f"- **{label}**: {reason}")
    
    with tab2:
        st.header("View Previous Taxonomies")
        st.markdown("Browse and explore previously generated taxonomies.")
        
        try:
            # Fetch all taxonomies from the database
            taxonomies = db_models.get_taxonomies()
            
            if taxonomies is None:
                st.warning("⚠️ Database connection issues detected. Trying to load taxonomies from files...")
                
                # Attempt to load from files as fallback
                try:
                    file_taxonomies = []
                    taxonomy_dir = Path('taxonomies')
                    if taxonomy_dir.exists() and taxonomy_dir.is_dir():
                        for file in taxonomy_dir.glob('*.json'):
                            try:
                                with open(file, 'r') as f:
                                    taxonomy_data = json.load(f)
                                    # Add file path as ID and source information
                                    taxonomy_data['id'] = str(file)
                                    taxonomy_data['source'] = 'file'
                                    file_taxonomies.append(taxonomy_data)
                            except Exception as file_err:
                                print(f"Error loading taxonomy file {file}: {file_err}")
                        
                        if file_taxonomies:
                            st.success(f"✅ Found {len(file_taxonomies)} taxonomies in files.")
                            taxonomies = file_taxonomies
                        else:
                            st.info("No taxonomy files found. Generate one in the 'Generate Taxonomy' tab.")
                    else:
                        st.info("No taxonomy directory found. Generate a taxonomy first.")
                except Exception as file_err:
                    st.error(f"Error reading taxonomy files: {file_err}")
                    taxonomies = []
            elif not taxonomies:
                st.info("No taxonomies found in the database. Generate one in the 'Generate Taxonomy' tab.")
            else:
                st.success(f"✅ Found {len(taxonomies)} taxonomies in the database.")
                
                # Create a dropdown to select which taxonomy to view
                taxonomy_options = [f"{t['domain']} ({t['timestamp']})" for t in taxonomies]
                selected_taxonomy_index = st.selectbox(
                    "Select a taxonomy to view",
                    range(len(taxonomy_options)),
                    format_func=lambda i: taxonomy_options[i]
                )
                
                if selected_taxonomy_index is not None:
                    selected_taxonomy = taxonomies[selected_taxonomy_index]
                    
                    # Display the selected taxonomy details
                    st.subheader(f"Taxonomy for '{selected_taxonomy['domain']}'")
                    st.markdown(f"**Generated on:** {selected_taxonomy['timestamp']}")
                    
                    # Display API provider if available (for backward compatibility with older taxonomies)
                    if 'api_provider' in selected_taxonomy:
                        st.markdown(f"**API Provider:** {selected_taxonomy['api_provider']}")
                    
                    st.markdown(f"**Tier-A Model:** {selected_taxonomy['tier_a_model']}")
                    st.markdown(f"**Tier-B Model:** {selected_taxonomy['tier_b_model']}")
                    
                    # Display approved labels
                    st.success(f"✅ {len(selected_taxonomy['approved_labels'])} Approved Labels:")
                    for label in selected_taxonomy['approved_labels']:
                        st.write(f"- {label}")
                    
                    # Display rejected labels and reasons
                    if selected_taxonomy['rejected_labels']:
                        with st.expander(f"⛔ {len(selected_taxonomy['rejected_labels'])} Rejected Labels"):
                            rejected_labels = selected_taxonomy['rejected_labels']
                            rejection_reasons = selected_taxonomy.get('rejection_reasons', {})
                            
                            for label in rejected_labels:
                                reason = rejection_reasons.get(label, "No reason provided")
                                st.write(f"- **{label}**: {reason}")
                    
                    # Add options to export/delete
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Export as JSON", key=f"export_{selected_taxonomy['id']}"):
                            # Create download link for the taxonomy
                            export_data = json.dumps(selected_taxonomy, indent=2)
                            st.download_button(
                                label="Download JSON", 
                                data=export_data,
                                file_name=f"{selected_taxonomy['domain'].replace(' ', '_')}.json",
                                mime="application/json"
                            )
                    
                    with col2:
                        if st.button("Delete Taxonomy", key=f"delete_{selected_taxonomy['id']}"):
                            if db_models.delete_taxonomy(selected_taxonomy['id']):
                                st.success(f"Taxonomy for '{selected_taxonomy['domain']}' deleted successfully.")
                                st.rerun()
                            else:
                                st.error("Failed to delete taxonomy.")
        except Exception as e:
            st.error(f"Error accessing taxonomy database: {e}")
    
    with tab3:
        st.header("Model Information")
        st.markdown("This page provides information about different AI models and recommendations for taxonomy generation.")
        
        model_tabs = st.tabs(["OpenAI Models", "Perplexity Models"])
        
        with model_tabs[0]:
            # OpenAI Models
            try:
                with open("model_info.md", "r") as f:
                    model_info = f.read()
                    st.markdown(model_info)
            except FileNotFoundError:
                st.info("OpenAI model information file not found.")
                # Provide basic information if file is missing
                st.markdown("""
                ## Recommended OpenAI Models
                
                - **For general use**: gpt-4o or gpt-3.5-turbo
                - **For advanced reasoning**: o1 or o3 (if you have access)
                
                Note that o-series models (o1, o3) may return empty responses depending on your account permissions.
                """)
        
        with model_tabs[1]:
            # Perplexity Models
            st.markdown("""
            ## Perplexity AI Models
            
            Perplexity offers several specialized model categories:
            
            ### Search Models
            
            #### sonar
            - Lightweight, cost-effective search model
            - Good for simple factual queries and basic information retrieval
            - Faster response times
            
            #### sonar-pro
            - Advanced search with grounding capabilities
            - Supports complex queries and follow-ups
            - More comprehensive information retrieval
            
            ### Research Models
            
            #### sonar-deep-research
            - Expert-level research model
            - Conducts exhaustive searches and generates comprehensive reports
            - Ideal for in-depth analysis with exhaustive web research
            
            ### Reasoning Models
            
            #### sonar-reasoning
            - Fast, real-time reasoning model
            - Designed for quick problem-solving with search
            - Recommended for taxonomy refinement tasks
            
            #### sonar-reasoning-pro
            - Premier reasoning model powered by DeepSeek R1
            - Features Chain of Thought (CoT) capabilities
            - Best for complex analyses requiring step-by-step thinking
            
            ### Offline Models
            
            #### r1-1776
            - DeepSeek R1 version for unconnected, unbiased, factual responses
            - No web search capabilities
            - Ideal for creative content generation without search interference
            
            ### Notes on Perplexity models:
            - Search models typically have an "-online" suffix when using web search
            - Models with "reasoning" in their name are optimized for structured taxonomy tasks
            - Response format is standardized across all models
            """)
        


if __name__ == "__main__":
    main()
