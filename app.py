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
import call_apis  # Use standard API implementation
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
DEFAULT_MAX_LABELS: int = 15
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
                      openai_api_key: Optional[str] = None, perplexity_api_key: Optional[str] = None,
                      tier_a_prompt_id: Optional[str] = None, tier_b_prompt_id: Optional[str] = None,
                      use_custom_prompts: bool = False):
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
        tier_a_prompt_id: ID of custom Tier-A prompt to use (if use_custom_prompts is True)
        tier_b_prompt_id: ID of custom Tier-B prompt to use (if use_custom_prompts is True)
        use_custom_prompts: Whether to use custom prompts from the database
        
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
    # Handle custom prompts for Tier-A if selected
    if use_custom_prompts and tier_a_prompt_id:
        try:
            # Get the selected custom prompt
            tier_a_prompt_data = db_models.get_custom_prompt(tier_a_prompt_id)
            if tier_a_prompt_data:
                # Replace template variables in the custom prompt
                prompt_A = tier_a_prompt_data["content"]
                prompt_A = prompt_A.replace("{{domain}}", domain)
                prompt_A = prompt_A.replace("{{max_labels}}", str(max_labels))
                prompt_A = prompt_A.replace("{{min_labels}}", str(min_labels))
                prompt_A = prompt_A.replace("{{deny_list}}", ", ".join(deny_list))
                
                st.info(f"🔹 Using custom Tier-A prompt: {tier_a_prompt_data['name']}")
            else:
                st.warning(f"Custom Tier-A prompt (ID: {tier_a_prompt_id}) not found. Using default prompt.")
                use_custom_prompts = False  # Fall back to default
        except Exception as e:
            st.error(f"Error loading custom Tier-A prompt: {e}")
            use_custom_prompts = False  # Fall back to default
    
    # Use default prompts if not using custom ones
    if not use_custom_prompts or not tier_a_prompt_id:
        if api_provider == "OpenAI":
            prompt_A = f"""
You are a domain taxonomy generator specializing in discrete events.

Domain: {domain}

TASK:
Generate a list of {min_labels}–{max_labels} distinct, top-level (L1) categories representing *specific types of events* or *discrete occurrences* within the '{domain}' domain. Think incidents, launches, breakthroughs, failures, breaches, discoveries, major releases, regulatory actions, etc.

Rules for Labels:
1. Format: TitleCase, 1–4 words. May include one internal hyphen (e.g., Model-Launch, Data-Breach, Regulatory-Approval). Start with a capital letter. NO hash symbols (#).
2. Event-Driven: Must describe *what happened* (an event, a change), not an ongoing state, capability, technology area, or general theme.
3. Specificity: Prefer specific event types over overly broad categories.
4. Exclusion: Avoid generic business terms like {', '.join(deny_list)}. These are handled separately.
5. Output Format: Return ONLY a JSON array of strings. Example: ["Model-Launch", "System-Outage", "Major-Discovery"]

Generate the JSON array now.
"""
        else:  # Perplexity
            # Create prompt for Perplexity
            prompt_A = call_perplexity_api.create_taxonomy_prompt(domain, max_labels, min_labels, deny_list)
    
    # Now call the appropriate API with the selected/custom prompt
    st.info(f"🔹 Generating Tier‑A candidates via {api_provider} API...")
    with st.spinner(f"Waiting for {api_provider} API response for generation..."):
        candidates: List[str] = []
        if api_provider == "OpenAI":
            # Now returns tuple of (processed_content, raw_content, timestamp)
            resp_A_processed, resp_A_raw, tier_a_timestamp = call_apis.call_tier_a_api(prompt_A, openai_api_key, tier_a_model)
        else:  # Perplexity
            # Now returns tuple of (processed_content, raw_content, timestamp)
            resp_A_processed, resp_A_raw, tier_a_timestamp = call_perplexity_api.call_perplexity_api_tier_a(
                prompt_A, perplexity_api_key, tier_a_model
            )
        resp_A = resp_A_processed  # Use the processed content for compatibility with existing code

    if resp_A:
        # Display raw response in expander
        with st.expander("Raw Tier-A Response"):
            st.code(str(resp_A), language="json")  # Convert to string for display

        import re  # Import re here to ensure it's available in this scope
        
        # HANDLE DIFFERENT RESPONSE TYPES
        # Case 1: List object response
        if isinstance(resp_A, list):
            try:
                candidates_raw = resp_A
                candidates = [str(c).strip().lstrip('# ') for c in candidates_raw if c is not None]
                st.success(f"✅ Tier-A proposed {len(candidates)} labels (API returned list)")
            except Exception as e:
                st.error(f"Error processing list response: {e}")
                return None, None, None
                
        # Case 2: String response with JSON array
        elif isinstance(resp_A, str):
            # Try to find JSON array in string
            json_match = re.search(r'\[.*?\]', resp_A, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                # Found JSON-like pattern
                try:
                    json_str = json_match.group(0)
                    candidates_raw = json.loads(json_str)
                    candidates = [str(c).strip().lstrip('# ') for c in candidates_raw if c is not None]
                    st.success(f"✅ Tier-A proposed {len(candidates)} labels (extracted JSON)")
                except Exception as e:
                    # Try to parse the whole response as JSON
                    try:
                        candidates_raw = json.loads(resp_A)
                        if isinstance(candidates_raw, list):
                            candidates = [str(c).strip().lstrip('# ') for c in candidates_raw if c is not None]
                            st.success(f"✅ Tier-A proposed {len(candidates)} labels (full JSON parse)")
                        else:
                            # Maybe the JSON has a nested array
                            for key, value in candidates_raw.items():
                                if isinstance(value, list) and len(value) > 0:
                                    candidates = [str(c).strip().lstrip('# ') for c in value if c is not None]
                                    st.success(f"✅ Tier-A proposed {len(candidates)} labels (from JSON key '{key}')")
                                    break
                            else:
                                st.error(f"Tier-A returned JSON without a labels array")
                                return None, None, None
                    except json.JSONDecodeError:
                        # Last resort: Line-by-line parsing
                        lines = [line.strip().lstrip('- ').lstrip('* ').lstrip('# ') 
                                for line in resp_A.split('\n') 
                                if line.strip() and not line.strip().startswith('```')]
                        
                        if lines and len(lines) > 3 and all(1 <= len(line.split()) <= 10 for line in lines):
                            candidates = lines
                            st.warning(f"Tier-A did not return valid JSON, using {len(candidates)} lines as labels")
                        else:
                            st.error(f"Tier-A response could not be parsed as labels")
                            return None, None, None
            else:
                # No JSON structure found, try line-by-line parsing
                lines = [line.strip().lstrip('- ').lstrip('* ').lstrip('# ') 
                        for line in resp_A.split('\n') 
                        if line.strip() and not line.strip().startswith('```')]
                
                if lines and len(lines) > 3 and all(1 <= len(line.split()) <= 10 for line in lines):
                    candidates = lines
                    st.warning(f"Tier-A did not return JSON, but parsed {len(candidates)} lines as potential labels")
                else:
                    st.error(f"Tier-A did not return a recognizable array or list")
                    return None, None, None
        
        # Case 3: Unknown type
        else:
            st.error(f"Tier-A returned an unsupported response type: {type(resp_A)}")
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
        # Handle custom prompts for Tier-B if selected
        if use_custom_prompts and tier_b_prompt_id:
            try:
                # Get the selected custom prompt
                tier_b_prompt_data = db_models.get_custom_prompt(tier_b_prompt_id)
                if tier_b_prompt_data:
                    # Replace template variables in the custom prompt
                    prompt_B = tier_b_prompt_data["content"]
                    prompt_B = prompt_B.replace("{{domain}}", domain)
                    prompt_B = prompt_B.replace("{{candidates_json}}", json.dumps(candidates))
                    prompt_B = prompt_B.replace("{{max_labels}}", str(max_labels))
                    prompt_B = prompt_B.replace("{{min_labels}}", str(min_labels))
                    prompt_B = prompt_B.replace("{{deny_list}}", ", ".join(deny_list))
                    
                    st.info(f"🔹 Using custom Tier-B prompt: {tier_b_prompt_data['name']}")
                else:
                    st.warning(f"Custom Tier-B prompt (ID: {tier_b_prompt_id}) not found. Using default prompt.")
                    use_custom_prompts = False  # Fall back to default
            except Exception as e:
                st.error(f"Error loading custom Tier-B prompt: {e}")
                use_custom_prompts = False  # Fall back to default
        
        # Use default prompts if not using custom ones
        if not use_custom_prompts or not tier_b_prompt_id:
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
4. Consolidation & Target Count: Merge clear synonyms or overly similar event types. Aim for a final list of {min_labels} (±1) distinct, high-value event categories. Prioritize the most significant and common event types for the domain.
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
            else:  # Perplexity
                # Create prompt for Perplexity - pass model name to determine output format
                prompt_B = call_perplexity_api.create_taxonomy_audit_prompt(
                    domain, candidates, max_labels, min_labels, deny_list, model_name=tier_b_selected_model
                )
        
        # Now call the appropriate API with the selected/custom prompt
        st.info(f"🔹 Refining taxonomy via {api_provider} API...")
        if api_provider == "OpenAI":
            with st.spinner(f"Waiting for {api_provider} API response for refinement..."):
                # Now returns tuple of (processed_content, raw_content, timestamp)
                audit_response_processed, audit_response_raw, tier_b_timestamp = call_apis.call_openai_api(
                    prompt_B, openai_api_key, tier_b_selected_model
                )
                audit_response_str = audit_response_processed  # Use processed content for compatibility
                
        else:  # Perplexity
            with st.spinner(f"Waiting for {api_provider} API response for refinement..."):
                # Now returns tuple of (processed_content, raw_content, timestamp)
                audit_response_processed, audit_response_raw, tier_b_timestamp = call_perplexity_api.call_perplexity_api_tier_b(
                    prompt_B, perplexity_api_key, tier_b_selected_model
                )
                audit_response_str = audit_response_processed  # Use processed content for compatibility

        if audit_response_str:
            # Display raw response in expander
            with st.expander("Raw Tier-B Response"):
                st.code(audit_response_str, language="json")

            # Check if this is a reasoning model response (text format)
            is_reasoning_model = "reasoning" in tier_b_selected_model.lower() if tier_b_selected_model else False
            
            if is_reasoning_model:
                # First, try post-processing with sonar to extract structured data
                st.info("Processing natural language output from reasoning model...")
                
                # Ensure we have the regex module imported
                import re
                
                # Use sonar to extract structured data from the natural language response
                structured_data = call_perplexity_api.extract_structured_data_with_sonar(
                    audit_response_str, 
                    perplexity_api_key
                )
                
                if structured_data and isinstance(structured_data, dict):
                    # Successfully extracted structured data
                    approved = structured_data.get("approved", [])
                    rejected = structured_data.get("rejected", [])
                    rejected_info = structured_data.get("reason_rejected", {})
                    
                    st.success(f"✅ Successfully extracted structured data: {len(approved)} approved labels, {len(rejected)} rejected labels")
                    
                    # Show the structured data in an expander
                    with st.expander("Extracted Structured Data"):
                        st.json(structured_data)
                        
                else:
                    # Fallback to regex-based extraction
                    try:
                        st.warning("Failed to extract with sonar. Falling back to regex parsing...")
                        
                        # Extract sections
                        approved_section_match = re.search(r'APPROVED LABELS:(.*?)(?:REJECTED LABELS:|$)', 
                                                        audit_response_str, re.DOTALL | re.IGNORECASE)
                        rejected_section_match = re.search(r'REJECTED LABELS:(.*?)(?:REJECTION REASONS:|$)', 
                                                        audit_response_str, re.DOTALL | re.IGNORECASE)
                        reasons_section_match = re.search(r'REJECTION REASONS:(.*?)$', 
                                                        audit_response_str, re.DOTALL | re.IGNORECASE)
                        
                        # Process approved labels
                        if approved_section_match:
                            approved_text = approved_section_match.group(1).strip()
                            approved = [line.strip() for line in approved_text.split('\n') 
                                        if line.strip() and not line.strip().startswith('#')]
                        else:
                            st.warning("Could not find APPROVED LABELS section. Using Tier-A candidates.")
                            approved = candidates
                        
                        # Process rejected labels
                        if rejected_section_match:
                            rejected_text = rejected_section_match.group(1).strip()
                            rejected = [line.strip() for line in rejected_text.split('\n') 
                                        if line.strip() and not line.strip().startswith('#')]
                        else:
                            rejected = []
                        
                        # Process rejection reasons
                        rejected_info = {}
                        if reasons_section_match:
                            reasons_text = reasons_section_match.group(1).strip()
                            reason_lines = [line.strip() for line in reasons_text.split('\n') 
                                            if line.strip() and not line.strip().startswith('#')]
                            
                            for line in reason_lines:
                                # Try to extract label and reason from lines like "Label: Reason"
                                parts = line.split(':', 1)
                                if len(parts) == 2:
                                    label = parts[0].strip()
                                    reason = parts[1].strip()
                                    rejected_info[label] = reason
                        
                        st.success(f"✅ Tier-B approved {len(approved)} labels using regex extraction.")
                        
                    except Exception as e:
                        st.error(f"Error parsing structured text from reasoning model: {e}")
                        st.expander("Error Details").exception(e)
                        approved = candidates  # Fallback to original candidates
            
            else:
                # JSON parsing for non-reasoning models
                try:
                    # First attempt - direct JSON parsing
                    try:
                        audit_result = json.loads(audit_response_str)
                    except json.JSONDecodeError:
                        # If direct parsing fails, try to extract JSON object pattern
                        st.warning("Direct JSON parsing failed. Attempting to extract JSON object from response.")
                        # Look for JSON object pattern in the response {....}
                        import re  # Ensure re is available in this scope
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
    
    # Initialize variables that might not be defined in all code paths
    resp_A_raw = locals().get('resp_A_raw', None)  # Get from locals if defined, else None
    audit_response_raw = locals().get('audit_response_raw', None)
    tier_a_timestamp = locals().get('tier_a_timestamp', None)
    tier_b_timestamp = locals().get('tier_b_timestamp', None)
    
    # Save to database with improved error handling and raw outputs
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
        api_provider=api_provider,
        tier_a_raw_output=resp_A_raw,
        tier_b_raw_output=audit_response_raw,
        tier_a_timestamp=tier_a_timestamp,
        tier_b_timestamp=tier_b_timestamp,
        tier_a_prompt_id=tier_a_prompt_id,
        tier_b_prompt_id=tier_b_prompt_id
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Generate Taxonomy", "View Previous Taxonomies", "Model Info", "Debug Prompts", "Prompt Editor"])
    
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
                # Custom prompts section
                st.subheader("Custom Prompts")
                
                # Check if we have custom prompts in the database
                has_custom_prompts = False
                try:
                    all_prompts = db_models.get_custom_prompts(api_provider=api_provider)
                    if all_prompts:
                        has_custom_prompts = True
                except Exception as e:
                    st.warning(f"Could not check for custom prompts: {e}")
                
                # Option to use custom prompts
                use_custom_prompts = st.checkbox(
                    "Use Custom Prompts", 
                    value=False,
                    help="Use custom prompts from the Prompt Editor instead of the default prompts",
                    disabled=not has_custom_prompts
                )
                
                if use_custom_prompts:
                    if not has_custom_prompts:
                        st.warning("No custom prompts found. Create some in the Prompt Editor tab first.")
                    else:
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            # Get Tier-A prompts
                            tier_a_prompts = db_models.get_custom_prompts(tier="A", api_provider=api_provider)
                            tier_a_prompt_options = [(p["name"], p["id"]) for p in tier_a_prompts]
                            
                            # Dropdown for Tier-A prompt
                            selected_tier_a_name, selected_tier_a_id = st.selectbox(
                                "Tier-A Prompt",
                                options=tier_a_prompt_options,
                                index=0,
                                format_func=lambda x: x[0],  # Display just the name
                                key="generate_tier_a_prompt_selector",
                                help="Select which Tier-A prompt version to use"
                            )
                        
                        with col_b:
                            # Get Tier-B prompts
                            tier_b_prompts = db_models.get_custom_prompts(tier="B", api_provider=api_provider)
                            tier_b_prompt_options = [(p["name"], p["id"]) for p in tier_b_prompts]
                            
                            # Dropdown for Tier-B prompt
                            selected_tier_b_name, selected_tier_b_id = st.selectbox(
                                "Tier-B Prompt",
                                options=tier_b_prompt_options,
                                index=0,
                                format_func=lambda x: x[0],  # Display just the name
                                key="generate_tier_b_prompt_selector",
                                help="Select which Tier-B prompt version to use"
                            )
                
                # Hidden fields to store custom prompt IDs
                st.text_input(
                    "Tier-A Prompt ID", 
                    value=selected_tier_a_id if use_custom_prompts and 'selected_tier_a_id' in locals() else "", 
                    key="tier_a_prompt_id_hidden", 
                    label_visibility="collapsed"
                )
                st.text_input(
                    "Tier-B Prompt ID", 
                    value=selected_tier_b_id if use_custom_prompts and 'selected_tier_b_id' in locals() else "", 
                    key="tier_b_prompt_id_hidden", 
                    label_visibility="collapsed"
                )
                st.text_input(
                    "Use Custom Prompts", 
                    value=str(use_custom_prompts), 
                    key="use_custom_prompts_hidden", 
                    label_visibility="collapsed"
                )
                
                st.markdown("---")
                st.subheader("Model Selection")
                
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
                        "Tier-A Target Labels", 
                        min_value=5, 
                        max_value=20, 
                        value=DEFAULT_MAX_LABELS,
                        help="Target number of labels for candidate generation (Tier-A)"
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
                        "Tier-B Final Labels", 
                        min_value=3, 
                        max_value=12, 
                        value=DEFAULT_MIN_LABELS,
                        help="Target number of labels for the final refined taxonomy (Tier-B)"
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
                
                # Check for custom prompts from hidden fields
                use_custom_prompts = st.session_state.get("use_custom_prompts_hidden", "").lower() == "true"
                tier_a_prompt_id = st.session_state.get("tier_a_prompt_id_hidden", "")
                tier_b_prompt_id = st.session_state.get("tier_b_prompt_id_hidden", "")
                
                # Generate taxonomy based on selected API provider
                approved, rejected, rejection_reasons = generate_taxonomy(
                    domain=domain,
                    tier_a_model=tier_a_model,
                    tier_b_model=tier_b_model,
                    max_labels=max_labels,
                    min_labels=min_labels,  # Now using the user-selected min_labels
                    deny_list=deny_list,
                    out_dir=out_dir,
                    api_provider=form_api_provider,
                    openai_api_key=openai_api_key,
                    perplexity_api_key=perplexity_api_key,
                    tier_a_prompt_id=tier_a_prompt_id if tier_a_prompt_id else None,
                    tier_b_prompt_id=tier_b_prompt_id if tier_b_prompt_id else None,
                    use_custom_prompts=use_custom_prompts
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
                    
                    # Display information about custom prompts used (if any)
                    if 'tier_a_prompt_id' in selected_taxonomy and selected_taxonomy['tier_a_prompt_id']:
                        try:
                            tier_a_prompt = db_models.get_custom_prompt(selected_taxonomy['tier_a_prompt_id'])
                            if tier_a_prompt:
                                st.info(f"📝 Custom Tier-A Prompt: {tier_a_prompt['name']}")
                        except Exception as e:
                            st.warning(f"Could not load Tier-A prompt info: {e}")
                    
                    if 'tier_b_prompt_id' in selected_taxonomy and selected_taxonomy['tier_b_prompt_id']:
                        try:
                            tier_b_prompt = db_models.get_custom_prompt(selected_taxonomy['tier_b_prompt_id'])
                            if tier_b_prompt:
                                st.info(f"📝 Custom Tier-B Prompt: {tier_b_prompt['name']}")
                        except Exception as e:
                            st.warning(f"Could not load Tier-B prompt info: {e}")
                            
                    # Display raw API outputs if available (new feature)
                    if 'tier_a_raw_output' in selected_taxonomy and selected_taxonomy['tier_a_raw_output']:
                        with st.expander("Tier-A Raw Output"):
                            st.code(selected_taxonomy['tier_a_raw_output'], language="json")
                            if 'tier_a_timestamp' in selected_taxonomy and selected_taxonomy['tier_a_timestamp']:
                                st.info(f"API call timestamp: {selected_taxonomy['tier_a_timestamp']}")
                    
                    if 'tier_b_raw_output' in selected_taxonomy and selected_taxonomy['tier_b_raw_output']:
                        with st.expander("Tier-B Raw Output"):
                            st.code(selected_taxonomy['tier_b_raw_output'], language="json")
                            if 'tier_b_timestamp' in selected_taxonomy and selected_taxonomy['tier_b_timestamp']:
                                st.info(f"API call timestamp: {selected_taxonomy['tier_b_timestamp']}")
                    
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
        
    with tab4:
        st.header("Debug Prompts")
        st.markdown("View the exact prompts used in the taxonomy generation pipeline for debugging purposes.")
        
        # Examples of the prompts used in each API provider
        api_provider_for_debug = st.selectbox(
            "API Provider",
            options=API_PROVIDERS,
            index=0,
            key="debug_api_provider",
            help="Select which API provider's prompts to display"
        )
        
        # Debug form for entering a test domain and settings
        with st.expander("Test Settings", expanded=True):
            debug_domain = st.text_input("Test Domain", value="Cybersecurity", key="debug_domain")
            debug_max_labels = st.slider("Tier-A Target Labels", min_value=5, max_value=15, value=DEFAULT_MAX_LABELS, key="debug_max_labels")
            debug_min_labels = st.slider("Tier-B Final Labels", min_value=3, max_value=12, value=DEFAULT_MIN_LABELS, key="debug_min_labels")
            debug_deny_list = st.text_area("Deny List (one term per line)", value=DEFAULT_DENY_LIST, key="debug_deny_list", height=100)
            
            # Process deny list
            debug_deny_set = set(line.strip() for line in debug_deny_list.split('\n') if line.strip())
        
        # Display the prompts
        st.subheader(f"{api_provider_for_debug} Prompts")
        
        # Tier-A Prompt
        st.markdown("### Tier-A Prompt (Candidate Generation)")
        if api_provider_for_debug == "OpenAI":
            tier_a_prompt = f"""
You are a domain taxonomy generator specializing in discrete events.

Domain: {debug_domain}

TASK:
Generate a list of 12–15 distinct, top-level (L1) categories representing *specific types of events* or *discrete occurrences* within the '{debug_domain}' domain. Think incidents, launches, breakthroughs, failures, breaches, discoveries, major releases, regulatory actions, etc.

Rules for Labels:
1. Format: TitleCase, 1–4 words. May include one internal hyphen (e.g., Model-Launch, Data-Breach, Regulatory-Approval). Start with a capital letter. NO hash symbols (#).
2. Event-Driven: Must describe *what happened* (an event, a change), not an ongoing state, capability, technology area, or general theme.
3. Specificity: Prefer specific event types over overly broad categories.
4. Exclusion: Avoid generic business terms like {', '.join(debug_deny_set)}. These are handled separately.
5. Output Format: Return ONLY a JSON array of strings. Example: ["Model-Launch", "System-Outage", "Major-Discovery"]

Generate the JSON array now.
"""
        else:  # Perplexity
            # Use the globally imported call_perplexity_api module
            tier_a_prompt = call_perplexity_api.create_taxonomy_prompt(debug_domain, debug_max_labels, debug_min_labels, debug_deny_set)
        
        st.code(tier_a_prompt, language="text")
        
        # Sample list of candidates (for demonstration purposes)
        sample_candidates = [
            "Data-Breach", 
            "Malware-Attack", 
            "Vulnerability-Disclosure", 
            "System-Outage", 
            "Security-Patch", 
            "Regulatory-Compliance", 
            "Authentication-Failure", 
            "Zero-Day-Exploit", 
            "Ransomware-Incident", 
            "DDoS-Attack"
        ]
        
        # Tier-B Prompt
        st.markdown("### Tier-B Prompt (Refinement)")
        if api_provider_for_debug == "OpenAI":
            tier_b_prompt = f"""
You are a meticulous taxonomy auditor enforcing specific principles.

Candidate Event Labels for Domain '{debug_domain}':
{json.dumps(sample_candidates, indent=2)}

Your Task:
Review the candidate labels based on the following principles and return a refined list.

Principles to Enforce:
1. Event-Driven Focus: Each label MUST represent a discrete event, incident, change, or occurrence. Reject labels describing general themes, capabilities, technologies, or ongoing states (e.g., "Machine Learning", "Cloud Infrastructure").
2. Formatting: Ensure labels are 1–4 words, TitleCase. Hyphens are allowed ONLY between words (e.g., "Data-Breach" is okay, "AI-Powered" as an event type might be questionable unless it refers to a specific *launch* event). No leading symbols like '#'.
3. Deny List: Reject any label containing the exact terms: {', '.join(debug_deny_set)}.
4. Consolidation & Target Count: Merge clear synonyms or overly similar event types. Aim for a final list of {debug_min_labels} (±1) distinct, high-value event categories. Prioritize the most significant and common event types for the domain.
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
        else:  # Perplexity
            # Use the globally imported call_perplexity_api module
            tier_b_prompt = call_perplexity_api.create_taxonomy_audit_prompt(
                debug_domain, sample_candidates, debug_max_labels, debug_min_labels, debug_deny_set, 
                model_name="sonar-reasoning"  # Default reasoning model for example
            )
        
        st.code(tier_b_prompt, language="text")
        
        # Additional notes section
        st.markdown("### Notes")
        st.info("""
        - The actual prompts used during taxonomy generation will use the exact domain, settings, and candidates from your input.
        - The sample candidates list above is for demonstration purposes only.
        - When using Perplexity's reasoning models, additional processing is applied to extract structured data from the natural language responses.
        """)
    
    with tab5:
        st.header("Prompt Editor")
        st.markdown("Create, edit, and test custom prompt versions for different taxonomy generation steps.")
        
        # Add clear instructions for using the prompt editor
        st.info("""
        ### How to Use the Prompt Editor
        
        1. **Select an API Provider** at the top (OpenAI or Perplexity)
        2. **Choose a prompt to view** from the dropdown in either column
        3. **The default prompts are read-only** - to make changes, click "Save as New Version"
        4. After creating a custom version, you can edit and update it as needed
        5. Use the "Test Prompt" section at the bottom to verify your custom prompts work correctly
        6. Your custom prompts will appear as options in the "Advanced Settings" of the Generate tab
        """)
        
        # First, let's initialize default prompts if needed
        if 'prompt_init_run' not in st.session_state:
            try:
                # Check if we have system prompts in the database
                system_prompts = db_models.get_custom_prompts()
                has_system_prompts = any(prompt.get("is_system") for prompt in system_prompts)
                
                if not has_system_prompts:
                    st.info("Initializing default prompts... This will only happen once.")
                    # Use the initialize_default_prompts.py script
                    import initialize_default_prompts
                    with st.spinner("Initializing default prompts (one-time setup)..."):
                        initialize_default_prompts.main()
                    st.success("Default prompt templates have been loaded.")
                
                # Mark initialization as complete
                st.session_state.prompt_init_run = True
            except Exception as e:
                st.error(f"Error initializing default prompts: {e}")
                st.session_state.prompt_init_run = False
        
        # API provider selection for prompts
        prompt_api_provider = st.selectbox(
            "API Provider",
            options=API_PROVIDERS,
            index=0,
            key="prompt_editor_api_provider",
            help="Select which API provider's prompts to view and edit"
        )
        
        # Create two columns for the two tiers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tier-A Prompts (Candidate Generation)")
            
            # Get all Tier-A prompts for the selected provider
            tier_a_prompts = db_models.get_custom_prompts(tier="A", api_provider=prompt_api_provider)
            
            # Extract names and IDs for the dropdown
            tier_a_prompt_options = [(p["name"], p["id"]) for p in tier_a_prompts]
            if not tier_a_prompt_options:
                tier_a_prompt_options = [("No prompts found", None)]
            
            # Default to the first option
            default_index = 0
            
            # Dropdown to select which prompt version to view/edit
            tier_a_selected_name, tier_a_selected_id = st.selectbox(
                "Select Prompt Version",
                options=tier_a_prompt_options,
                index=default_index,
                format_func=lambda x: x[0],  # Display just the name
                key="tier_a_prompt_selector"
            )
            
            # If no prompt is selected or no prompts exist
            if tier_a_selected_id is None:
                st.warning("No Tier-A prompts found for this provider. Initialize the app to create default prompts.")
                tier_a_content = ""
                tier_a_description = ""
                is_system_a = False
            else:
                # Get the selected prompt details
                selected_prompt_a = db_models.get_custom_prompt(tier_a_selected_id)
                if selected_prompt_a:
                    tier_a_content = selected_prompt_a["content"]
                    tier_a_description = selected_prompt_a.get("description", "")
                    is_system_a = selected_prompt_a.get("is_system", False)
                else:
                    tier_a_content = ""
                    tier_a_description = ""
                    is_system_a = False
            
            # Display whether this is a system prompt
            if is_system_a:
                st.info("This is a system prompt (read-only). To make changes, save as a new version.")
            
            # Text area for prompt content (read-only if it's a system prompt)
            tier_a_edited_content = st.text_area(
                "Prompt Content",
                value=tier_a_content,
                height=300,
                disabled=is_system_a,
                key="tier_a_prompt_content"
            )
            
            # Description field (read-only if it's a system prompt)
            tier_a_edited_description = st.text_area(
                "Description",
                value=tier_a_description,
                height=100,
                disabled=is_system_a,
                key="tier_a_prompt_description",
                help="Add notes about what this prompt version does differently"
            )
            
            # Buttons for Tier-A prompt actions
            a_col1, a_col2, a_col3 = st.columns(3)
            
            with a_col1:
                # Save as new version button
                if st.button("Save as New Version", key="tier_a_save_new"):
                    new_name = f"Custom {prompt_api_provider} Tier-A Prompt {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    new_id = db_models.create_custom_prompt(
                        name=new_name,
                        tier="A",
                        api_provider=prompt_api_provider,
                        content=tier_a_edited_content,
                        description=tier_a_edited_description,
                        is_system=False
                    )
                    if new_id:
                        st.success(f"New prompt version created with ID: {new_id}")
                        # Refresh to show the new version in the dropdown
                        st.rerun()
                    else:
                        st.error("Failed to create new prompt version.")
            
            with a_col2:
                # Update button (only enabled for non-system prompts)
                update_disabled = is_system_a or tier_a_selected_id is None
                if st.button("Update Current", key="tier_a_update", disabled=update_disabled):
                    if db_models.update_custom_prompt(
                        prompt_id=tier_a_selected_id,
                        content=tier_a_edited_content,
                        description=tier_a_edited_description
                    ):
                        st.success("Prompt updated successfully.")
                    else:
                        st.error("Failed to update prompt.")
            
            with a_col3:
                # Delete button (only enabled for non-system prompts)
                delete_disabled = is_system_a or tier_a_selected_id is None
                if st.button("Delete Version", key="tier_a_delete", disabled=delete_disabled):
                    if db_models.delete_custom_prompt(tier_a_selected_id):
                        st.success("Prompt deleted successfully.")
                        # Refresh to update the dropdown
                        st.rerun()
                    else:
                        st.error("Failed to delete prompt.")
        
        with col2:
            st.subheader("Tier-B Prompts (Refinement)")
            
            # Get all Tier-B prompts for the selected provider
            tier_b_prompts = db_models.get_custom_prompts(tier="B", api_provider=prompt_api_provider)
            
            # Extract names and IDs for the dropdown
            tier_b_prompt_options = [(p["name"], p["id"]) for p in tier_b_prompts]
            if not tier_b_prompt_options:
                tier_b_prompt_options = [("No prompts found", None)]
            
            # Default to the first option
            default_index = 0
            
            # Dropdown to select which prompt version to view/edit
            tier_b_selected_name, tier_b_selected_id = st.selectbox(
                "Select Prompt Version",
                options=tier_b_prompt_options,
                index=default_index,
                format_func=lambda x: x[0],  # Display just the name
                key="tier_b_prompt_selector"
            )
            
            # If no prompt is selected or no prompts exist
            if tier_b_selected_id is None:
                st.warning("No Tier-B prompts found for this provider. Initialize the app to create default prompts.")
                tier_b_content = ""
                tier_b_description = ""
                is_system_b = False
            else:
                # Get the selected prompt details
                selected_prompt_b = db_models.get_custom_prompt(tier_b_selected_id)
                if selected_prompt_b:
                    tier_b_content = selected_prompt_b["content"]
                    tier_b_description = selected_prompt_b.get("description", "")
                    is_system_b = selected_prompt_b.get("is_system", False)
                else:
                    tier_b_content = ""
                    tier_b_description = ""
                    is_system_b = False
            
            # Display whether this is a system prompt
            if is_system_b:
                st.info("This is a system prompt (read-only). To make changes, save as a new version.")
            
            # Text area for prompt content (read-only if it's a system prompt)
            tier_b_edited_content = st.text_area(
                "Prompt Content",
                value=tier_b_content,
                height=300,
                disabled=is_system_b,
                key="tier_b_prompt_content"
            )
            
            # Description field (read-only if it's a system prompt)
            tier_b_edited_description = st.text_area(
                "Description",
                value=tier_b_description,
                height=100,
                disabled=is_system_b,
                key="tier_b_prompt_description",
                help="Add notes about what this prompt version does differently"
            )
            
            # Buttons for Tier-B prompt actions
            b_col1, b_col2, b_col3 = st.columns(3)
            
            with b_col1:
                # Save as new version button
                if st.button("Save as New Version", key="tier_b_save_new"):
                    new_name = f"Custom {prompt_api_provider} Tier-B Prompt {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    new_id = db_models.create_custom_prompt(
                        name=new_name,
                        tier="B",
                        api_provider=prompt_api_provider,
                        content=tier_b_edited_content,
                        description=tier_b_edited_description,
                        is_system=False
                    )
                    if new_id:
                        st.success(f"New prompt version created with ID: {new_id}")
                        # Refresh to show the new version in the dropdown
                        st.rerun()
                    else:
                        st.error("Failed to create new prompt version.")
            
            with b_col2:
                # Update button (only enabled for non-system prompts)
                update_disabled = is_system_b or tier_b_selected_id is None
                if st.button("Update Current", key="tier_b_update", disabled=update_disabled):
                    if db_models.update_custom_prompt(
                        prompt_id=tier_b_selected_id,
                        content=tier_b_edited_content,
                        description=tier_b_edited_description
                    ):
                        st.success("Prompt updated successfully.")
                    else:
                        st.error("Failed to update prompt.")
            
            with b_col3:
                # Delete button (only enabled for non-system prompts)
                delete_disabled = is_system_b or tier_b_selected_id is None
                if st.button("Delete Version", key="tier_b_delete", disabled=delete_disabled):
                    if db_models.delete_custom_prompt(tier_b_selected_id):
                        st.success("Prompt deleted successfully.")
                        # Refresh to update the dropdown
                        st.rerun()
                    else:
                        st.error("Failed to delete prompt.")
        
        # Test section with sample configuration
        st.subheader("Test Prompt")
        
        with st.expander("Test Configuration", expanded=False):
            test_domain = st.text_input("Test Domain", value="Cybersecurity", key="prompt_test_domain")
            test_max_labels = st.slider("Tier-A Target Labels", min_value=5, max_value=15, value=DEFAULT_MAX_LABELS, key="prompt_test_max_labels")
            test_min_labels = st.slider("Tier-B Final Labels", min_value=3, max_value=12, value=DEFAULT_MIN_LABELS, key="prompt_test_min_labels")
            test_deny_list = st.text_area("Deny List (one term per line)", value=DEFAULT_DENY_LIST, key="prompt_test_deny_list")
            
            # Process deny list
            test_deny_set = set(line.strip() for line in test_deny_list.split('\n') if line.strip())
            
            # Tier selection for testing
            test_tier = st.radio("Test Which Tier?", options=["Tier-A", "Tier-B", "Both"], index=0, key="prompt_test_tier")
            
            # Model selection based on API provider
            if prompt_api_provider == "OpenAI":
                test_model_options = DEFAULT_OPENAI_TIER_A_OPTIONS if test_tier == "Tier-A" else DEFAULT_OPENAI_TIER_B_OPTIONS
                default_model = "gpt-3.5-turbo"  # A good default that works reliably
                
                # Custom model selection
                test_model = st.selectbox(
                    "Model to Test With",
                    options=test_model_options,
                    index=test_model_options.index(default_model) if default_model in test_model_options else 0,
                    key="prompt_test_model_openai"
                )
                
                # If custom is selected, show an input field
                if test_model == "custom":
                    test_model = st.text_input(
                        "Custom Model Name",
                        value="gpt-4o-mini",
                        key="prompt_test_custom_model_openai",
                        help="Enter a valid OpenAI model name"
                    )
            else:  # Perplexity
                if test_tier == "Tier-A":
                    test_model_options = DEFAULT_PERPLEXITY_TIER_A_OPTIONS
                    default_model = "sonar"
                else:
                    test_model_options = DEFAULT_PERPLEXITY_TIER_B_OPTIONS
                    default_model = "sonar-reasoning"
                
                # Custom model selection
                test_model = st.selectbox(
                    "Model to Test With",
                    options=test_model_options,
                    index=test_model_options.index(default_model) if default_model in test_model_options else 0,
                    key="prompt_test_model_perplexity"
                )
                
                # If custom is selected, show an input field
                if test_model == "custom":
                    test_model = st.text_input(
                        "Custom Model Name",
                        value="sonar-pro" if test_tier == "Tier-A" else "sonar-reasoning-pro",
                        key="prompt_test_custom_model_perplexity",
                        help="Enter a valid Perplexity model name"
                    )
            
            # Sample candidates for Tier-B testing
            if test_tier in ["Tier-B", "Both"]:
                test_candidates = st.text_area(
                    "Sample Candidates for Tier-B (one per line)", 
                    value="Vulnerability-Disclosure\nData-Breach\nRansomware-Attack\nSystem-Compromise\nPatch-Release\nZero-Day-Exploit\nMalware-Detection\nNetwork-Intrusion\nPhishing-Campaign\nIdentity-Theft",
                    key="prompt_test_candidates",
                    height=150
                )
                candidates_list = [c.strip() for c in test_candidates.split('\n') if c.strip()]
            else:
                candidates_list = []
            
            # API keys
            test_openai_key = st.text_input("OpenAI API Key (optional)", value="", key="prompt_test_openai_key", type="password")
            test_perplexity_key = st.text_input("Perplexity API Key (optional)", value="", key="prompt_test_perplexity_key", type="password")
            
            # Use environment variables if not provided
            if not test_openai_key:
                test_openai_key = openai_api_key
            if not test_perplexity_key:
                test_perplexity_key = perplexity_api_key
        
        # Test button
        col1, col2 = st.columns([1, 3])
        with col1:
            test_button = st.button("Test Prompt", key="prompt_test_button")
        
        if test_button:
            if test_tier == "Tier-A" or test_tier == "Both":
                st.subheader("Tier-A Test Results")
                if tier_a_selected_id is None:
                    st.error("Please select a valid Tier-A prompt to test.")
                else:
                    # Call the API with the selected prompt
                    st.info(f"Sending Tier-A prompt to {prompt_api_provider} API...")
                    
                    # Replace template variables in the prompt
                    formatted_prompt = tier_a_edited_content
                    formatted_prompt = formatted_prompt.replace("{{domain}}", test_domain)
                    formatted_prompt = formatted_prompt.replace("{{max_labels}}", str(test_max_labels))
                    formatted_prompt = formatted_prompt.replace("{{min_labels}}", str(test_min_labels))
                    formatted_prompt = formatted_prompt.replace("{{deny_list}}", ", ".join(test_deny_set))
                    
                    # Show the formatted prompt
                    with st.expander("Formatted Prompt"):
                        st.code(formatted_prompt, language="text")
                    
                    try:
                        if prompt_api_provider == "OpenAI":
                            with st.spinner(f"Calling OpenAI API with model {test_model}..."):
                                response, raw_response, timestamp = call_apis.call_tier_a_api(
                                    formatted_prompt, 
                                    test_openai_key,
                                    test_model
                                )
                        else:  # Perplexity
                            # Import at the global level to avoid local variable issues
                            import call_perplexity_api as cp_api
                            with st.spinner(f"Calling Perplexity API with model {test_model}..."):
                                response, raw_response, timestamp = cp_api.call_perplexity_api_tier_a(
                                    formatted_prompt, 
                                    test_perplexity_key,
                                    test_model
                                )
                        
                        if response:
                            st.success("✅ Tier-A prompt test successful!")
                            st.json(response)
                            candidates_list = response if isinstance(response, list) else []
                        else:
                            st.error("❌ API call succeeded but returned empty or invalid response.")
                            if raw_response:
                                st.expander("Raw API Response").code(raw_response, language="json")
                    except Exception as e:
                        st.error(f"❌ Error testing Tier-A prompt: {e}")
            
            if test_tier == "Tier-B" or test_tier == "Both":
                st.subheader("Tier-B Test Results")
                if tier_b_selected_id is None:
                    st.error("Please select a valid Tier-B prompt to test.")
                elif not candidates_list:
                    st.error("No candidate labels available for Tier-B testing.")
                else:
                    # Call the API with the selected prompt
                    st.info(f"Sending Tier-B prompt to {prompt_api_provider} API...")
                    
                    # Replace template variables in the prompt
                    formatted_prompt = tier_b_edited_content
                    formatted_prompt = formatted_prompt.replace("{{domain}}", test_domain)
                    formatted_prompt = formatted_prompt.replace("{{candidates_json}}", json.dumps(candidates_list))
                    formatted_prompt = formatted_prompt.replace("{{max_labels}}", str(test_max_labels))
                    formatted_prompt = formatted_prompt.replace("{{min_labels}}", str(test_min_labels))
                    formatted_prompt = formatted_prompt.replace("{{deny_list}}", ", ".join(test_deny_set))
                    
                    # Show the formatted prompt
                    with st.expander("Formatted Prompt"):
                        st.code(formatted_prompt, language="text")
                    
                    try:
                        if prompt_api_provider == "OpenAI":
                            with st.spinner(f"Calling OpenAI API with model {test_model}..."):
                                response, raw_response, timestamp = call_apis.call_openai_api(
                                    formatted_prompt, 
                                    test_openai_key,
                                    test_model
                                )
                        else:  # Perplexity
                            # Import at the global level to avoid local variable issues
                            import call_perplexity_api as cp_api
                            with st.spinner(f"Calling Perplexity API with model {test_model}..."):
                                response, raw_response, timestamp = cp_api.call_perplexity_api_tier_b(
                                    formatted_prompt, 
                                    test_perplexity_key,
                                    test_model
                                )
                                
                                # If this is a reasoning model response, try to extract structured data
                                if response and "sonar-reasoning" in test_model:
                                    structured_data = cp_api.extract_structured_data_with_sonar(
                                        response, 
                                        test_perplexity_key
                                    )
                                    if structured_data:
                                        response = structured_data
                        
                        if response:
                            st.success("✅ Tier-B prompt test successful!")
                            st.json(response)
                        else:
                            st.error("❌ API call succeeded but returned empty or invalid response.")
                            if raw_response:
                                st.expander("Raw API Response").code(raw_response, language="json")
                    except Exception as e:
                        st.error(f"❌ Error testing Tier-B prompt: {e}")

        # Help section
        st.markdown("---")
        st.markdown("### Prompt Template Variables")
        st.info("""
        Your prompt templates can include these variables which will be replaced with actual values:
        - `{{domain}}` - The domain for taxonomy generation
        - `{{max_labels}}` - Maximum number of labels (Tier-A target)
        - `{{min_labels}}` - Minimum number of labels (Tier-B target)
        - `{{deny_list}}` - Comma-separated list of denied terms
        - `{{candidates_json}}` - JSON array of candidate labels (for Tier-B only)
        """)


if __name__ == "__main__":
    main()
