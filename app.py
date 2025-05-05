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

# API clients
try:
    from openai import OpenAI, APIError as OpenAI_APIError, AuthenticationError as OpenAI_AuthError
    from openai import RateLimitError as OpenAI_RateLimitError, APIConnectionError as OpenAI_ConnError
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

# ----- Configuration (Defaults) -----
# OpenAI models for both tiers
# The newest OpenAI model is "gpt-4o" which was released May 13, 2024
DEFAULT_TIER_A_OPTIONS: List[str] = [
    "gpt-4o",  # Powerful, reliable model
    "gpt-3.5-turbo"  # Faster, more economical
]

DEFAULT_TIER_B_OPTIONS: List[str] = ["gpt-4o", "gpt-3.5-turbo", "None/Offline"]
DEFAULT_MAX_LABELS: int = 9
DEFAULT_MIN_LABELS: int = 8
DEFAULT_DENY_LIST: str = "Funding\nHiring\nPartnership"
DEFAULT_OUT_DIR: str = "taxonomies"

# ----- Setup Logging -----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- Helper Functions for API Calls -----
def call_tier_a_api(prompt: str, api_key: Optional[str], model_name: str) -> Optional[str]:
    """Calls the OpenAI API for Tier-A (candidate generation) with retry logic."""
    if not OPENAI_AVAILABLE:
        st.error("OpenAI library required for Tier-A call but not found/installed.")
        return None
    if not api_key:
        st.error("OPENAI_API_KEY required for Tier-A call but not found/set.")
        return None

    # Retry parameters
    max_retries = 3
    retry_delays = [2, 5, 10]  # Exponential backoff in seconds
    
    for attempt in range(max_retries + 1):
        try:
            client = OpenAI(api_key=api_key)
            
            if attempt > 0:
                st.info(f"🔄 Retry attempt {attempt}/{max_retries} for Tier-A (OpenAI) model...")
            else:
                st.info(f"🔹 Calling Tier-A (OpenAI) model ({model_name})...")
                
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,  # Generous buffer for JSON list
            )
            content = response.choices[0].message.content
            if content:
                if attempt > 0:
                    st.success(f"Successfully retrieved response after {attempt} retries.")
                return content.strip()
            else:
                st.error("Tier-A (OpenAI) API returned an empty response.")
                return None
                
        except OpenAI_RateLimitError:
            if attempt < max_retries:
                delay = retry_delays[attempt]
                st.warning(f"⚠️ Tier-A API rate limit hit. Waiting {delay} seconds before retry {attempt+1}/{max_retries}...")
                with st.spinner(f"Waiting {delay} seconds..."):
                    time.sleep(delay)
            else:
                st.error("Tier-A API Error (OpenAI): Rate limit exceeded after maximum retries.")
                st.info("This happens when too many requests are made to the OpenAI API in a short period. Please try again later or adjust your usage pattern.")
                return None
                
        except OpenAI_AuthError:
            st.error("Tier-A API Error (OpenAI): Authentication failed. Check your OPENAI_API_KEY.")
            return None
            
        except OpenAI_ConnError:
            if attempt < max_retries:
                delay = retry_delays[attempt]
                st.warning(f"⚠️ Connection error. Waiting {delay} seconds before retry {attempt+1}/{max_retries}...")
                with st.spinner(f"Waiting {delay} seconds..."):
                    time.sleep(delay)
            else:
                st.error("Tier-A API Error (OpenAI): Could not connect to OpenAI API after maximum retries.")
                return None
                
        except OpenAI_APIError as e:
            if attempt < max_retries and ("502" in str(e) or "503" in str(e) or "504" in str(e)):
                delay = retry_delays[attempt]
                st.warning(f"⚠️ API error ({e}). Waiting {delay} seconds before retry {attempt+1}/{max_retries}...")
                with st.spinner(f"Waiting {delay} seconds..."):
                    time.sleep(delay)
            else:
                st.error(f"Tier-A API Error (OpenAI): {e}")
                return None
                
        except Exception as e:
            st.error(f"An unexpected error occurred during Tier-A (OpenAI) call: {e}")
            return None


def call_openai_api(prompt: str, api_key: Optional[str], model_name: str) -> Optional[str]:
    """Calls the OpenAI API (Tier-B) with retry logic."""
    if model_name.lower() == "none/offline":
        st.warning("Tier‑B model call skipped (selected None/Offline).")
        return None
    if not OPENAI_AVAILABLE:
        st.error("OpenAI library required for Tier-B call but not found/installed.")
        return None
    if not api_key:
        st.error("OPENAI_API_KEY required for Tier-B call but not found/set.")
        return None

    # Retry parameters
    max_retries = 3
    retry_delays = [2, 5, 10]  # Exponential backoff in seconds
    
    for attempt in range(max_retries + 1):
        try:
            client = OpenAI(api_key=api_key)
            
            if attempt > 0:
                st.info(f"🔄 Retry attempt {attempt}/{max_retries} for Tier-B (OpenAI) model...")
            else:
                st.info(f"🔹 Calling Tier‑B (OpenAI) model ({model_name})...")
                
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"}  # Request JSON
            )
            content = response.choices[0].message.content
            if content:
                if attempt > 0:
                    st.success(f"Successfully retrieved response after {attempt} retries.")
                return content.strip()
            else:
                st.error("Tier-B (OpenAI) API returned an empty response.")
                return None
                
        except OpenAI_RateLimitError:
            if attempt < max_retries:
                delay = retry_delays[attempt]
                st.warning(f"⚠️ Tier-B API rate limit hit. Waiting {delay} seconds before retry {attempt+1}/{max_retries}...")
                with st.spinner(f"Waiting {delay} seconds..."):
                    time.sleep(delay)
            else:
                st.error("Tier-B API Error (OpenAI): Rate limit exceeded after maximum retries.")
                st.info("This happens when too many requests are made to the OpenAI API in a short period. Please try again later or adjust your usage pattern.")
                return None
                
        except OpenAI_AuthError:
            st.error("Tier-B API Error (OpenAI): Authentication failed. Check your OPENAI_API_KEY.")
            return None
            
        except OpenAI_ConnError:
            if attempt < max_retries:
                delay = retry_delays[attempt]
                st.warning(f"⚠️ Connection error. Waiting {delay} seconds before retry {attempt+1}/{max_retries}...")
                with st.spinner(f"Waiting {delay} seconds..."):
                    time.sleep(delay)
            else:
                st.error("Tier-B API Error (OpenAI): Could not connect to OpenAI API after maximum retries.")
                return None
                
        except OpenAI_APIError as e:
            if attempt < max_retries and ("502" in str(e) or "503" in str(e) or "504" in str(e)):
                delay = retry_delays[attempt]
                st.warning(f"⚠️ API error ({e}). Waiting {delay} seconds before retry {attempt+1}/{max_retries}...")
                with st.spinner(f"Waiting {delay} seconds..."):
                    time.sleep(delay)
            else:
                st.error(f"Tier-B API Error (OpenAI): {e}")
                return None
                
        except Exception as e:
            st.error(f"An unexpected error occurred during Tier-B (OpenAI) call: {e}")
            return None


# ----- Core Taxonomy Generation Logic -----
def generate_taxonomy(domain: str, tier_a_model: str, tier_b_model: str, max_labels: int, min_labels: int, 
                      deny_list: set, out_dir: Path, openai_api_key: Optional[str]):
    """The main function to generate and validate the taxonomy using APIs."""
    
    if not domain:
        st.error("Domain input cannot be empty.")
        return None, None, None

    st.info(f"Processing domain: {domain}")
    out_dir.mkdir(exist_ok=True)

    # ----- Tier‑A candidate generation (OpenAI API) -----
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

    st.info("🔹 Generating Tier‑A candidates via OpenAI API...")
    with st.spinner("Waiting for OpenAI API response for generation..."):
        candidates: List[str] = []
        resp_A = call_tier_a_api(prompt_A, openai_api_key, tier_a_model)

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
        return None, None, None

    # Display the candidates
    st.subheader("Tier-A Candidates")
    st.write(candidates)

    # ----- Tier‑B audit & refinement (OpenAI API) -----
    approved: List[str] = []
    rejected: List[str] = []
    rejected_info: Dict[str, str] = {}
    tier_b_selected_model = tier_b_model

    if tier_b_selected_model.lower() != "none/offline":
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
        with st.spinner("Waiting for OpenAI API response for refinement..."):
            audit_response_str = call_openai_api(prompt_B, openai_api_key, tier_b_selected_model)

        if audit_response_str:
            # Display raw response in expander
            with st.expander("Raw Tier-B Response"):
                st.code(audit_response_str, language="json")

            try:
                audit_result = json.loads(audit_response_str)
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
                    approved = candidates
            except json.JSONDecodeError:
                st.error("Tier-B returned unparsable JSON.")
                approved = candidates  # Fallback to using all candidates
        else:
            st.warning("No Tier-B refinement performed. Using Tier-A candidates as final.")
            approved = candidates
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
    
    # Save to database
    try:
        taxonomy_id = db_models.create_taxonomy(
            domain=domain,
            tier_a_model=tier_a_model,
            tier_b_model=tier_b_model,
            max_labels=max_labels,
            min_labels=min_labels,
            deny_list=deny_list,
            approved_labels=approved,
            rejected_labels=rejected if rejected else [],
            rejection_reasons=rejected_info if rejected_info else {}
        )
        st.success(f"Taxonomy saved to database with ID: {taxonomy_id}")
    except Exception as e:
        st.error(f"Failed to save taxonomy to database: {e}")

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
    This app helps you generate taxonomies for any domain using a two-tier approach:
    - **Tier-A** (OpenAI API): Generates candidate labels
    - **Tier-B** (OpenAI API): Refines and validates the taxonomy
    
    Use the tabs below to generate a new taxonomy or view previously generated ones.
    """)
    
    # API Key warnings/status (outside tabs)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        st.success("✅ OPENAI_API_KEY found in environment variables")
    else:
        st.error("❌ OPENAI_API_KEY not found. Set it in your environment variables.")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Generate Taxonomy", "View Previous Taxonomies"])
    
    with tab1:
        st.header("Generate New Taxonomy")
        st.markdown("Enter your domain and configuration parameters below to start.")

        # Input Form
        with st.form("taxonomy_config_form"):
            domain = st.text_input("Domain", help="Enter the domain for which you want to generate a taxonomy (e.g., 'Artificial Intelligence', 'Healthcare Tech')")
            
            # Advanced settings expander
            with st.expander("Advanced Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    tier_a_model = st.selectbox(
                        "Tier-A Model (OpenAI)", 
                        options=DEFAULT_TIER_A_OPTIONS,
                        index=0,
                        help="Select the OpenAI model for candidate generation"
                    )
                    
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
                    tier_b_model = st.selectbox(
                        "Tier-B Model (OpenAI)", 
                        options=DEFAULT_TIER_B_OPTIONS,
                        index=0,
                        help="Select the OpenAI model for refinement (or None/Offline to skip)"
                    )
                    
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
                # Generate taxonomy
                approved, rejected, rejection_reasons = generate_taxonomy(
                    domain=domain,
                    tier_a_model=tier_a_model,
                    tier_b_model=tier_b_model,
                    max_labels=max_labels,
                    min_labels=min_labels,
                    deny_list=deny_list,
                    out_dir=out_dir,
                    openai_api_key=openai_api_key
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
            
            if not taxonomies:
                st.info("No taxonomies found in the database. Generate one in the 'Generate Taxonomy' tab.")
            else:
                st.success(f"Found {len(taxonomies)} taxonomies in the database.")
                
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
            

if __name__ == "__main__":
    main()
