"""
Perplexity API integration for taxonomy generation

This module provides functions to call the Perplexity AI API as an alternative
to OpenAI for generating taxonomies. Perplexity offers more reliable results
with reasoning capabilities.
"""

import os
import re
import time
import json
import logging
import traceback
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime

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
    deny_list: set,
    model_name: str = ""
) -> str:
    """
    Create a prompt for Perplexity API to audit and refine a taxonomy.
    
    Args:
        domain: The domain for taxonomy generation
        candidates: List of candidate labels
        max_labels: Maximum number of labels
        min_labels: Minimum number of labels
        deny_list: Set of denied terms
        model_name: The model being used (to customize output format)
        
    Returns:
        str: Formatted prompt
    """
    denied_terms = ", ".join(deny_list) if deny_list else "none"
    
    # Check if this is a reasoning model
    is_reasoning_model = "reasoning" in model_name.lower() if model_name else False
    
    if is_reasoning_model:
        # For reasoning models, use a more natural language output format with explicit thinking tags
        prompt = f"""
You are a meticulous taxonomy auditor enforcing specific principles.

Candidate Event Labels for Domain '{domain}':
{json.dumps(candidates, indent=2)}

Your Task:
Review the candidate labels based on the following principles and return a refined list.

Principles to Enforce:
1. Event-Driven Focus: Each label MUST represent a discrete event, incident, change, or occurrence. Reject labels describing general themes, capabilities, technologies, or ongoing states.
2. Formatting: Labels must be 1–4 words, TitleCase, with hyphens between words (e.g., "Data-Breach").
3. Deny List: Reject any label containing the terms: {denied_terms}.
4. Consolidation & Target Count: Merge synonyms or similar event types. Aim for exactly {min_labels} labels (±1). This means between {min_labels-1} and {min_labels+1} labels.

5. Output Format: Use this EXACT format:
<thinking>
Your detailed analysis process here...
</thinking>

APPROVED LABELS:
[List each approved label, one per line]

REJECTED LABELS:
[List each rejected label, one per line]

REJECTION REASONS:
[For each rejected label: "Label: Reason for rejection"]

The <thinking> tags help me understand your reasoning process, but I'll extract only the final lists after those tags.
"""
    else:
        # For non-reasoning models, use the JSON output format
        prompt = f"""
You are a meticulous taxonomy auditor enforcing specific principles.

Candidate Event Labels for Domain '{domain}':
{json.dumps(candidates, indent=2)}

Your Task:
Review the candidate labels based on the following principles and return a refined list.

Principles to Enforce:
1. Event-Driven Focus: Each label MUST represent a discrete event, incident, change, or occurrence. Reject labels describing general themes, capabilities, technologies, or ongoing states.
2. Formatting: Labels must be 1–4 words, TitleCase, with hyphens between words (e.g., "Data-Breach").
3. Deny List: Reject any label containing the terms: {denied_terms}.
4. Consolidation & Target Count: Merge synonyms or similar event types. Aim for exactly {min_labels} labels (±1). This means between {min_labels-1} and {min_labels+1} labels.
5. Output Structure: Return ONLY a JSON object with the following keys:
   - "approved": Array of approved labels (aim for {min_labels} ±1 labels)
   - "rejected": Array of rejected labels
   - "reason_rejected": Object mapping each rejected label to a reason

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


def call_perplexity_api_tier_a(prompt: str, api_key: Optional[str], model_name: str = "sonar") -> Tuple[Union[List[str], str, None], Optional[str], Optional[datetime]]:
    """
    Call Perplexity API for Tier-A candidate generation.
    
    Args:
        prompt: Prompt for the API
        api_key: Perplexity API key
        model_name: Perplexity model to use
        
    Returns:
        Tuple containing:
        - str or None: Processed API response content or None on failure
        - str or None: Raw API response content for logging/debugging
        - datetime or None: Timestamp when the API was called
    """
    if not api_key:
        st.error("PERPLEXITY_API_KEY required but not found in environment variables.")
        return None, None, None
    
    # Check if key has proper format
    if not api_key.startswith("pplx-"):
        st.warning("Perplexity API key doesn't have the expected format (should start with 'pplx-')")
    
    # Note: Perplexity model names are used as-is without adding -online suffix
    # The online search capability is built into models like sonar and sonar-pro
    
    st.info(f"🔹 Calling Tier-A (Perplexity) model ({model_name})...")
    
    # Record the API call timestamp at the start
    api_timestamp = datetime.now()
    
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
        
        # Update API call timestamp after receiving response
        api_timestamp = datetime.now()
        
        # Extract content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                # Store the raw response for debugging/analysis
                raw_content = content
                
                # Try to process the content with our robust parser
                try:
                    # Use the flexible parser for Tier-A to handle various response formats
                    # This allows more flexible prompts that might return JSON, lists, or other structures
                    labels = flexible_response_parser(content)
                    
                    if labels and isinstance(labels, list) and len(labels) > 0:
                        # If we successfully extracted labels, return them
                        logger.info(f"Successfully extracted {len(labels)} labels using flexible parser")
                        return labels, raw_content, api_timestamp
                    
                    # If we couldn't extract labels but have content, return the content
                    if isinstance(content, str):
                        # This maintains backward compatibility with the existing processing logic
                        processed_content = content.strip()
                        return processed_content, raw_content, api_timestamp
                    elif isinstance(content, list):
                        # If it's already a list, return it directly
                        return content, json.dumps(content), api_timestamp
                    else:
                        # Convert to string as a last resort
                        return str(content), str(content), api_timestamp
                        
                except Exception as parse_error:
                    # Log the parsing error but continue with raw content
                    logger.error(f"Error in flexible parsing: {parse_error}")
                    logger.error(traceback.format_exc())
                    
                    # Return the raw content as a fallback
                    if isinstance(content, str):
                        return content, content, api_timestamp
                    elif isinstance(content, list):
                        return content, json.dumps(content), api_timestamp
                    else:
                        return str(content), str(content), api_timestamp
        
        st.error(f"Perplexity API returned an empty or invalid response for Tier-A.")
        return None, None, api_timestamp
        
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
        
        return None, None, api_timestamp


def extract_structured_data_from_text(text: Union[str, List, Any], api_key: Optional[str]) -> Optional[Dict]:
    """
    Use the sonar model to extract structured data from natural language output.
    
    Args:
        text: The natural language output to process
        api_key: Perplexity API key
        
    Returns:
        Dict or None: Structured data with approved/rejected labels and reasons
    """
    if not api_key:
        st.error("PERPLEXITY_API_KEY required but not found in environment variables.")
        return None
    
    st.info("🔍 Using sonar model to extract structured data from natural language output...")
    
    # Handle different input types for text
    text_to_process = text
    
    # Convert any non-string input to a string representation
    if not isinstance(text, str):
        logger.warning(f"extract_structured_data_from_text received non-string input: {type(text)}")
        try:
            if isinstance(text, list):
                text_to_process = "\n".join([str(item) for item in text])
                logger.info("Converted list to string for sonar extraction")
            elif isinstance(text, dict):
                text_to_process = json.dumps(text, indent=2)
                logger.info("Converted dict to JSON string for sonar extraction")
            else:
                text_to_process = str(text)
                logger.info(f"Converted {type(text)} to string for sonar extraction")
        except Exception as e:
            logger.error(f"Failed to convert input to string: {e}")
            st.error(f"Failed to process input data for extraction: {e}")
            return None
    
    prompt = f"""
Extract structured data from the following text, which contains a taxonomy review with approved labels, rejected labels, and rejection reasons:

'''
{text_to_process}
'''

Return this information as a valid JSON object with the following structure:
{{
  "approved": ["Label1", "Label2", "Label3", ...],
  "rejected": ["RejectedLabel1", "RejectedLabel2", ...],
  "reason_rejected": {{
    "RejectedLabel1": "Reason for rejection",
    "RejectedLabel2": "Reason for rejection",
    ...
  }}
}}

Make sure the approved and rejected lists are complete and accurate based on the text.
"""
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts structured data from text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        logger.info("Calling Perplexity API (sonar) for JSON extraction")
        
        response = client.chat.completions.create(
            model="sonar",  # Use regular sonar, not reasoning models
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            top_p=1,
            response_format={"type": "json_object"}  # Request JSON format
        )
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                try:
                    # Parse the JSON response
                    structured_data = json.loads(content.strip())
                    return structured_data
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse JSON from sonar extraction: {e}")
                    st.expander("Raw Extraction Result").code(content)
                    return None
        
        st.error("Sonar returned an empty or invalid response for extraction.")
        return None
        
    except Exception as e:
        st.error(f"Error calling Perplexity sonar for extraction: {e}")
        return None


def call_perplexity_api_tier_b(prompt: str, api_key: Optional[str], model_name: str = "sonar-reasoning") -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    """
    Call Perplexity API for Tier-B refinement.
    
    Args:
        prompt: Prompt for the API
        api_key: Perplexity API key
        model_name: Perplexity model to use
        
    Returns:
        Tuple containing:
        - str or None: Processed API response content or None on failure
        - str or None: Raw API response content for logging/debugging
        - datetime or None: Timestamp when the API was called
    """
    if not api_key:
        st.error("PERPLEXITY_API_KEY required but not found in environment variables.")
        return None, None, None
    
    # Check if key has proper format
    if not api_key.startswith("pplx-"):
        st.warning("Perplexity API key doesn't have the expected format (should start with 'pplx-')")
    
    # Note: Perplexity model names are used as-is without adding -online suffix
    # The online search capability is built into models like sonar and sonar-pro
    
    st.info(f"🔹 Calling Tier-B (Perplexity) model ({model_name})...")
    
    # Record the API call timestamp at the start
    api_timestamp = datetime.now()
    
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
        
        # Update API call timestamp after receiving response
        api_timestamp = datetime.now()
        
        # Extract content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                # Add robust error handling for Tier-B responses as well
                try:
                    # For Tier-B, we primarily want the reasoning output for later extraction
                    # But we can still try the flexible parser for diagnostic purposes
                    labels = flexible_response_parser(content)
                    if labels and isinstance(labels, list):
                        logger.info(f"Flexible parser found {len(labels)} potential labels in Tier-B response")
                    
                    # Determine return type based on content type
                    if isinstance(content, str):
                        # Store both the processed content and raw response
                        processed_content = content.strip()
                        raw_content = content  # Store the raw response for debugging/analysis
                        return processed_content, raw_content, api_timestamp
                    elif isinstance(content, list):
                        # If content is a list, convert to JSON for the raw content
                        processed_content = json.dumps(content)
                        raw_content = json.dumps(content)
                        return processed_content, raw_content, api_timestamp
                    else:
                        # For any other type, convert to string
                        processed_content = str(content)
                        raw_content = str(content)
                        return processed_content, raw_content, api_timestamp
                
                except Exception as parse_error:
                    # Log the error but continue with simple string conversion
                    logger.error(f"Error in Tier-B response processing: {parse_error}")
                    logger.error(traceback.format_exc())
                    
                    # Fallback to simple string conversion
                    try:
                        processed_content = str(content)
                        raw_content = str(content)
                        return processed_content, raw_content, api_timestamp
                    except:
                        # Ultimate fallback
                        return "Error processing response", "Error processing response", api_timestamp
        
        st.error(f"Perplexity API returned an empty or invalid response for Tier-B.")
        return None, None, api_timestamp
        
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
        
        return None, None, api_timestamp


def extract_structured_data_with_sonar(text: Union[str, List, Any], api_key: Optional[str]) -> Optional[Dict]:
    """
    Use Perplexity's sonar model to extract structured data from natural language responses.
    
    Args:
        text: Text response from a reasoning model
        api_key: Perplexity API key
        
    Returns:
        Dict or None: Structured data extracted from text
    """
    if not api_key:
        st.error("PERPLEXITY_API_KEY required but not found in environment variables.")
        return None
    
    # Handle different input types for text before creating the prompt
    text_to_process = text
    
    # Convert any non-string input to a string representation
    if not isinstance(text, str):
        logger.warning(f"extract_structured_data_with_sonar received non-string input: {type(text)}")
        try:
            if isinstance(text, list):
                text_to_process = "\n".join([str(item) for item in text])
                logger.info("Converted list to string for sonar processing")
            elif isinstance(text, dict):
                text_to_process = json.dumps(text, indent=2)
                logger.info("Converted dict to JSON string for sonar processing")
            else:
                text_to_process = str(text)
                logger.info(f"Converted {type(text)} to string for sonar processing")
        except Exception as e:
            logger.error(f"Failed to convert text to string for sonar: {e}")
            st.error(f"Failed to process input data: {e}")
            return None
    
    # Create a prompt to extract the structured data with the processed text
    prompt = f"""
Extract structured data from the following taxonomy evaluation text. Output ONLY a valid JSON object following the exact structure specified:

TEXT TO PROCESS:
```
{text_to_process}
```

REQUIRED OUTPUT FORMAT:
A JSON object with these three keys exactly as shown:
- "approved": An array of strings (the accepted labels)
- "rejected": An array of strings (the rejected labels)
- "reason_rejected": An object mapping each rejected label to its reason

EXAMPLE OUTPUT:
```json
{{
  "approved": ["Model-Launch", "System-Outage", "Regulatory-Action"],
  "rejected": ["AI Research", "Funding Round", "ProductUpdate"],
  "reason_rejected": {{
    "AI Research": "Not event-driven, describes a theme.",
    "Funding Round": "Contains denied term 'Funding'.",
    "ProductUpdate": "Merged into Major-Release."
  }}
}}
```

Extract and format the structure precisely. Return ONLY the JSON object with no other text.
"""
    
    try:
        st.info("Using sonar to extract structured data from reasoning model output...")
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        
        # Create system message and user message
        messages = [
            {
                "role": "system",
                "content": "You are a specialized data extraction tool that transforms natural language into structured JSON. Return ONLY valid JSON with no explanation, markdown, or other text. The output must be parsable by JSON.parse() directly."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Make API call to sonar model
        response = client.chat.completions.create(
            model="sonar",  # Using sonar for structure extraction
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            top_p=1
            # Perplexity API doesn't support the response_format parameter like OpenAI
        )
        
        # Extract content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                # Try to parse the extracted JSON
                try:
                    # First try direct JSON parsing
                    # The system message asks for JSON output
                    data = json.loads(content)
                    
                    # Verify the extracted structure
                    if (isinstance(data, dict) and 
                        "approved" in data and isinstance(data["approved"], list) and
                        "rejected" in data and isinstance(data["rejected"], list) and
                        "reason_rejected" in data and isinstance(data["reason_rejected"], dict)):
                        
                        st.success("✅ Successfully extracted structured data from reasoning model output")
                        return data
                    else:
                        st.warning("Extracted JSON has an invalid structure")
                        st.expander("Invalid Structure").json(data)
                except json.JSONDecodeError:
                    st.warning("Could not parse JSON from sonar extraction")
                    
                    # Fallback: Try to extract JSON pattern if direct parsing fails
                    try:
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            data = json.loads(json_str)
                            
                            if (isinstance(data, dict) and 
                                "approved" in data and isinstance(data["approved"], list) and
                                "rejected" in data and isinstance(data["rejected"], list) and
                                "reason_rejected" in data and isinstance(data["reason_rejected"], dict)):
                                
                                st.success("✅ Successfully extracted JSON object from response")
                                return data
                            else:
                                st.warning("Extracted JSON object has an invalid structure")
                        else:
                            st.warning("Could not find JSON object in sonar extraction response")
                    except Exception:
                        st.error("Failed to extract valid JSON from response")
        
        st.error("Sonar extraction failed to produce valid structured data")
        return None
        
    except Exception as e:
        st.error(f"Error using sonar for extraction: {e}")
        return None


def flexible_response_parser(text: Union[str, List, Any]) -> List[str]:
    """
    A flexible parser that can extract labels from various response formats.
    Works with plain lists, JSON, and structured text.
    
    Args:
        text: Raw text response from the API (can be string, list, or other)
        
    Returns:
        List of extracted labels
    """
    logger.info(f"Parsing response with flexible parser - type: {type(text)}")
    labels = []
    
    # Handle case where API directly returns a list
    if isinstance(text, list):
        logger.info("Input is already a list, processing directly")
        return [str(item).strip() for item in text if item is not None]
    
    # Ensure we're working with a string for text-based processing
    if not isinstance(text, str):
        logger.warning(f"Input is not a string or list but {type(text)}, converting to string")
        try:
            text = str(text)
        except Exception as e:
            logger.error(f"Failed to convert to string: {e}")
            return []
    
    # Strategy 1: Direct JSON parsing if it looks like JSON
    if text.strip().startswith('{') or text.strip().startswith('['):
        try:
            data = json.loads(text)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Direct list of labels
                labels = [str(item) for item in data if isinstance(item, (str, int))]
            elif isinstance(data, dict):
                # Try to find arrays in the dict that might contain labels
                for key, value in data.items():
                    if key in ["labels", "categories", "approved", "L1_categories", "root_domains"]:
                        if isinstance(value, list):
                            labels.extend([str(item) for item in value if isinstance(item, (str, int))])
            
            # If we found labels, return them
            if labels:
                logger.info(f"Extracted {len(labels)} labels from JSON")
                return labels
        except json.JSONDecodeError:
            logger.info("Not valid JSON, trying other methods")
    
    # Strategy 2: Line-by-line parsing for plain text lists
    lines = text.strip().split('\n')
    clean_lines = []
    
    for line in lines:
        # Strip common list prefixes and whitespace
        clean_line = re.sub(r'^[-*•#\d.)\s]+', '', line).strip()
        
        # Only include lines that look like labels (not empty, not too long)
        if clean_line and len(clean_line) < 100 and not clean_line.startswith('```'):
            clean_lines.append(clean_line)
    
    if clean_lines:
        logger.info(f"Extracted {len(clean_lines)} labels from line-by-line parsing")
        return clean_lines
    
    # Strategy 3: Extract items with patterns that look like labels or categories
    label_patterns = [
        r'["\']([A-Za-z0-9-]{3,50})["\']',  # Quoted labels
        r'<([A-Za-z0-9-]{3,50})>',          # Labels in angle brackets
        r'#([A-Za-z0-9-]{3,50})',           # Hashtag labels
        r'([A-Z][a-z]+-[A-Z][a-z]+(?:-[A-Z][a-z]+)?)'  # TitleCase-Words with hyphens
    ]
    
    for pattern in label_patterns:
        matches = re.findall(pattern, text)
        if matches:
            labels.extend(matches)
    
    if labels:
        # Remove duplicates while preserving order
        unique_labels = []
        seen = set()
        for label in labels:
            if label.lower() not in seen:
                seen.add(label.lower())
                unique_labels.append(label)
        
        logger.info(f"Extracted {len(unique_labels)} labels using regex patterns")
        return unique_labels
    
    # Fallback: Extract any words that might be labels (last resort)
    words = re.findall(r'\b([A-Z][a-z]{2,}(?:-[A-Z][a-z]{2,})*)\b', text)
    if words:
        # Deduplicate
        unique_words = list(set(words))
        logger.info(f"Extracted {len(unique_words)} possible label words as fallback")
        return unique_words
    
    logger.warning("Could not extract any labels with flexible parser")
    return []