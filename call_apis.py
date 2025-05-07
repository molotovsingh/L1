"""API calling functions for OpenAI models"""

import time
import logging
from typing import Optional, Tuple, List, Union
from datetime import datetime

import streamlit as st
from openai import OpenAI, APIError as OpenAI_APIError, AuthenticationError as OpenAI_AuthError
from openai import RateLimitError as OpenAI_RateLimitError, APIConnectionError as OpenAI_ConnError

import model_mapper

# Import the flexible parser from our new utility module
try:
    from src.utils.parsers import flexible_response_parser
except ImportError:
    # If the import fails, create a local copy of the function for backward compatibility
    import re
    import json
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def flexible_response_parser(text: str) -> List[str]:
        """
        A flexible parser that can extract labels from various response formats.
        Works with plain lists, JSON, and structured text.
        
        Args:
            text: Raw text response from the API
            
        Returns:
            List of extracted labels
        """
        logger.info("Parsing response with flexible parser")
        labels = []
        
        # Try multiple parsing strategies
        
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

def call_tier_a_api(prompt: str, api_key: Optional[str], model_name: str) -> Tuple[Union[List[str], str, None], Optional[str], Optional[datetime]]:
    """
    Calls the OpenAI API for Tier-A (candidate generation) with retry logic.
    
    Args:
        prompt: Prompt for the API
        api_key: OpenAI API key
        model_name: OpenAI model to use
        
    Returns:
        Tuple containing:
        - Union[List[str], str, None]: Processed response (list of labels, raw string, or None)
        - str or None: Raw API response content for logging/debugging
        - datetime or None: Timestamp when the API was called
    """
    if not api_key:
        st.error("OPENAI_API_KEY required for Tier-A call but not found/set.")
        return None, None, None
        
    # Map the model name (handle shorthands like o1, o3)
    model_name = model_mapper.map_model_name(model_name)

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
                
            # Prepare parameters based on model
            o_series_models = ["o1", "o3", "o4-mini", "o1-mini", "o3-mini", "o1-preview", "o1-pro"]
            is_o_series = model_name in o_series_models
            
            # Base parameters (will be adjusted for model type)
            params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,  # Generous buffer for JSON list
            }
            
            # Add temperature for non-o-series (o-series only supports default temp=1.0)
            if not is_o_series:
                params["temperature"] = 0.0
            
            # Apply model-specific parameter adjustments
            params = model_mapper.get_model_params(model_name, params)
            
            # For o-series models, modify the prompt to request clearer output
            if is_o_series:
                # Use system message to request structured output since response_format isn't available
                system_prompt = "You are an AI assistant focused on taxonomy generation. Generate labels as requested in a clear, structured format."
                params["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt + "\n\nImportant: Provide a clear list of labels."}
                ]
            
            # Record the timestamp when the API was called
            api_timestamp = datetime.now()
            
            # Make the API call with the adjusted parameters
            response = client.chat.completions.create(**params)
            
            # Debug response for o-series models
            if is_o_series:
                # Create an expandable section for the debug info
                with st.expander("Debug - o-series API response details:"):
                    st.info(f"Model: {model_name}")
                    st.info(f"Response object type: {type(response)}")
                    st.info(f"Choices available: {len(response.choices)}")
                    
                    if len(response.choices) > 0:
                        msg = response.choices[0].message
                        st.info(f"Message role: {msg.role}")
                        st.info(f"Content length: {len(msg.content) if msg.content else 0}")
                        if not msg.content:
                            st.error("Content is empty or None")
                        elif len(msg.content) < 100:
                            st.code(f"{msg.content}")
                        else:
                            st.code(f"{msg.content[:100]}...")
            
            # Process response normally
            if len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    if attempt > 0:
                        st.success(f"Successfully retrieved response after {attempt} retries.")
                    # Preserve raw content for database storage
                    processed_content = content.strip()
                    raw_content = content
                    return processed_content, raw_content, api_timestamp
            
            # For o-series models, if we get empty response, try a fallback approach
            if is_o_series:
                st.warning("Received empty response from o-series model, trying fallback approach...")
                # Simplify the prompt for o-series models
                simpler_prompt = "Generate a list of important taxonomy labels for the domain. Just list the labels directly."
                fallback_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Keep your response simple and direct."},
                        {"role": "user", "content": simpler_prompt}
                    ],
                    "max_completion_tokens": params.get("max_completion_tokens", 512)
                }
                
                try:
                    fallback_response = client.chat.completions.create(**fallback_params)
                    if len(fallback_response.choices) > 0 and fallback_response.choices[0].message.content:
                        fallback_content = fallback_response.choices[0].message.content
                        st.success("Got response with fallback approach!")
                        return fallback_content.strip()
                except Exception as fallback_e:
                    st.error(f"Fallback approach also failed: {fallback_e}")
            
            # If we reached here, we couldn't get a valid response
            st.error("Tier-A (OpenAI) API returned an empty response.")
            
            if is_o_series:
                st.warning("""
                ⚠️ **O-series Model Failed**
                
                The o-series model attempted did not return usable content. This is a common limitation of these models
                with certain account permissions. For more reliable results, please try:
                
                1. Use standard GPT models like `gpt-4o` or `gpt-3.5-turbo` instead
                2. Check the "Model Info" tab for detailed information about model capabilities
                """)
            return None
                
        except OpenAI_RateLimitError as e:
            error_msg = str(e).lower()
            if "insufficient_quota" in error_msg or "quota" in error_msg:
                st.error("OpenAI API Error: Account quota exceeded")
                st.warning("⚠️ **OpenAI API Quota Exhausted**")
                st.info("""
                Your OpenAI API key has exceeded its quota limit.
                
                **Recommended solutions:**
                1. Check your billing details at OpenAI.com
                2. Update your payment method if needed
                3. Consider upgrading your OpenAI API plan
                4. Or use a different API key with available quota
                """)
                return None
                
            # Regular rate limiting (too many requests in short time)
            if attempt < max_retries:
                delay = retry_delays[attempt]
                st.warning(f"⚠️ Tier-A API rate limit hit. Waiting {delay} seconds before retry {attempt+1}/{max_retries}...")
                with st.spinner(f"Waiting {delay} seconds..."):
                    time.sleep(delay)
            else:
                st.error("Tier-A API Error (OpenAI): Rate limit exceeded after maximum retries.")
                st.warning("⚠️ **OpenAI API Rate Limit Reached**")
                st.info("""
                This happens when too many requests are made to the OpenAI API in a short period.
                
                **Recommended solutions:**
                1. Wait 1-2 minutes before trying again
                2. Try using a different model (gpt-3.5-turbo has higher rate limits)
                3. If the problem persists, you may need to wait up to an hour for quota reset
                """)
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


def call_openai_api(prompt: str, api_key: Optional[str], model_name: str) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    """
    Calls the OpenAI API (Tier-B) with retry logic.
    
    Args:
        prompt: Prompt for the API
        api_key: OpenAI API key
        model_name: OpenAI model to use
        
    Returns:
        Tuple containing:
        - str or None: Processed API response content or None on failure
        - str or None: Raw API response content for logging/debugging
        - datetime or None: Timestamp when the API was called
    """
    if model_name.lower() == "none/offline":
        st.warning("Tier‑B model call skipped (selected None/Offline).")
        return None, None, None
    if not api_key:
        st.error("OPENAI_API_KEY required for Tier-B call but not found/set.")
        return None, None, None
        
    # Map the model name (handle shorthands like o1, o3)
    model_name = model_mapper.map_model_name(model_name)

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
                
            # Prepare parameters based on model
            o_series_models = ["o1", "o3", "o4-mini", "o1-mini", "o3-mini", "o1-preview", "o1-pro"]
            is_o_series = model_name in o_series_models
            
            # Base parameters (will be adjusted for model type)
            params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
            }
            
            # Add temperature for non-o-series (o-series only supports default temp=1.0)
            if not is_o_series:
                params["temperature"] = 0.0
            
            # Add response_format for models that support it
            # Note: Not all o-series models support structured JSON output formats
            if not is_o_series:
                params["response_format"] = {"type": "json_object"}  # Request JSON
            
            # Apply model-specific parameter adjustments
            params = model_mapper.get_model_params(model_name, params)
            
            # For o-series models, modify the prompt to request JSON-like output
            if is_o_series:
                # Use system message to request structured output since response_format isn't available
                system_prompt = "You are an AI assistant focused on taxonomy refinement. Respond only with valid JSON in the requested format. No explanations or extra text."
                params["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt + "\n\nImportant: Respond ONLY with valid JSON. No explanations or other text."}
                ]
            
            # Record the timestamp when the API was called
            api_timestamp = datetime.now()
            
            # Make the API call with the adjusted parameters
            response = client.chat.completions.create(**params)
            
            # Debug response for o-series models
            if is_o_series:
                # Create an expandable section for the debug info
                with st.expander("Debug - o-series API response details (Tier-B):"):
                    st.info(f"Model: {model_name}")
                    st.info(f"Response object type: {type(response)}")
                    st.info(f"Choices available: {len(response.choices)}")
                    
                    if len(response.choices) > 0:
                        msg = response.choices[0].message
                        st.info(f"Message role: {msg.role}")
                        st.info(f"Content length: {len(msg.content) if msg.content else 0}")
                        if not msg.content:
                            st.error("Content is empty or None")
                        elif len(msg.content) < 100:
                            st.code(f"{msg.content}")
                        else:
                            st.code(f"{msg.content[:100]}...")
            
            # Process response normally
            if len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    if attempt > 0:
                        st.success(f"Successfully retrieved response after {attempt} retries.")
                    # Preserve raw content for database storage
                    processed_content = content.strip()
                    raw_content = content
                    return processed_content, raw_content, api_timestamp
            
            # For o-series models, if we get empty response, try a fallback approach
            if is_o_series:
                st.warning("Received empty response from o-series model (Tier-B), trying fallback approach...")
                # Simplify the prompt for o-series models
                fallback_prompt = prompt
                if "JSON" in prompt:
                    fallback_prompt += "\n\nIf you have any difficulty producing valid JSON, just respond with your recommendations in a simple list format."
                    
                fallback_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Keep your response simple and direct."},
                        {"role": "user", "content": fallback_prompt}
                    ],
                    "max_completion_tokens": params.get("max_completion_tokens", 512)
                }
                
                try:
                    fallback_response = client.chat.completions.create(**fallback_params)
                    if len(fallback_response.choices) > 0 and fallback_response.choices[0].message.content:
                        fallback_content = fallback_response.choices[0].message.content
                        st.success("Got response with fallback approach!")
                        return fallback_content.strip()
                except Exception as fallback_e:
                    st.error(f"Fallback approach also failed: {fallback_e}")
            
            # If we reached here, we couldn't get a valid response
            st.error("Tier-B (OpenAI) API returned an empty response.")
            
            if is_o_series:
                st.warning("""
                ⚠️ **O-series Model Failed**
                
                The o-series model attempted did not return usable content. This is a common limitation of these models
                with certain account permissions. For more reliable results, please try:
                
                1. Use standard GPT models like `gpt-4o` or `gpt-3.5-turbo` instead
                2. Check the "Model Info" tab for detailed information about model capabilities
                """)
            return None
                
        except OpenAI_RateLimitError as e:
            error_msg = str(e).lower()
            if "insufficient_quota" in error_msg or "quota" in error_msg:
                st.error("OpenAI API Error: Account quota exceeded")
                st.warning("⚠️ **OpenAI API Quota Exhausted**")
                st.info("""
                Your OpenAI API key has exceeded its quota limit.
                
                **Recommended solutions:**
                1. Check your billing details at OpenAI.com
                2. Update your payment method if needed
                3. Consider upgrading your OpenAI API plan
                4. Or use a different API key with available quota
                """)
                return None
                
            # Regular rate limiting (too many requests in short time)
            if attempt < max_retries:
                delay = retry_delays[attempt]
                st.warning(f"⚠️ Tier-B API rate limit hit. Waiting {delay} seconds before retry {attempt+1}/{max_retries}...")
                with st.spinner(f"Waiting {delay} seconds..."):
                    time.sleep(delay)
            else:
                st.error("Tier-B API Error (OpenAI): Rate limit exceeded after maximum retries.")
                st.warning("⚠️ **OpenAI API Rate Limit Reached**")
                st.info("""
                This happens when too many requests are made to the OpenAI API in a short period.
                
                **Recommended solutions:**
                1. Wait 1-2 minutes before trying again
                2. Try using a different model (gpt-3.5-turbo has higher rate limits)
                3. If the problem persists, you may need to wait up to an hour for quota reset
                """)
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