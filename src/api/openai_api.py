"""
OpenAI API integration for taxonomy generation.

This module provides functions to interact with OpenAI's API
for generating taxonomies.
"""

import os
import re
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

import streamlit as st

# Import from the new package structure
from src.utils.parsers import flexible_response_parser

try:
    from openai import OpenAI, APIError as OpenAI_APIError
    from openai import RateLimitError as OpenAI_RateLimitError
    from openai import AuthenticationError as OpenAI_AuthError
    from openai import APIConnectionError as OpenAI_ConnError
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False
    OpenAI_APIError = Exception
    OpenAI_RateLimitError = Exception
    OpenAI_AuthError = Exception
    OpenAI_ConnError = Exception

# Global settings for API retry logic
MAX_RETRIES = 4
RETRY_DELAYS = [5, 15, 30, 60]  # Exponential backoff in seconds

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment."""
    return os.environ.get("OPENAI_API_KEY")

def call_tier_a_api(prompt: str, api_key: Optional[str], model_name: str) -> Tuple[Union[List[str], str, None], Optional[str], Optional[datetime]]:
    """
    Calls the OpenAI API for Tier-A (candidate generation) with retry logic.
    
    Args:
        prompt: Prompt for the API
        api_key: OpenAI API key
        model_name: OpenAI model to use
        
    Returns:
        Tuple containing:
        - Union[List[str], str, None]: Processed response (list of labels, raw content, or None)
        - str or None: Raw API response content for logging/debugging
        - datetime or None: Timestamp when the API was called
    """
    if not api_key:
        api_key = get_openai_api_key()
        if not api_key:
            st.error("❌ OpenAI API key is missing. Please check your environment variables.")
            return None, None, None
    
    timestamp = datetime.now()
    st.info(f"🔹 Calling Tier-A (OpenAI) with model {model_name}...")
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Format message for the API
    messages = [
        {"role": "system", "content": "You are a domain taxonomy generator specializing in structured taxonomies."},
        {"role": "user", "content": prompt}
    ]
    
    # API call with retry logic
    for retry_count in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                top_p=1
            )
            
            # If we get here, the API call succeeded
            timestamp = datetime.now()
            
            if hasattr(response, 'choices') and len(response.choices) > 0:
                raw_response = response.choices[0].message.content
                
                if raw_response:
                    # First try to use our flexible parser to extract structured data
                    extracted_labels = flexible_response_parser(raw_response)
                    if extracted_labels:
                        return extracted_labels, raw_response, timestamp
                    
                    # If no structured data could be extracted, return the raw text
                    return raw_response, raw_response, timestamp
                else:
                    st.warning(f"⚠️ Model {model_name} returned empty content")
                    return None, None, timestamp
            else:
                st.error(f"❌ Error: API response missing choices")
                return None, None, timestamp
            
        except OpenAI_RateLimitError as e:
            if retry_count < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[retry_count]
                st.warning(f"⚠️ Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error(f"❌ Rate limit exceeded after {MAX_RETRIES} retries.")
                return None, str(e), timestamp
                
        except OpenAI_APIError as e:
            st.error(f"❌ API Error: {str(e)}")
            return None, str(e), timestamp
            
        except OpenAI_AuthError as e:
            st.error(f"❌ Authentication Error: Please check your API key.")
            return None, str(e), timestamp
            
        except OpenAI_ConnError as e:
            if retry_count < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[retry_count]
                st.warning(f"⚠️ Connection error. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error(f"❌ Connection error after {MAX_RETRIES} retries.")
                return None, str(e), timestamp
                
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None, str(e), timestamp
            
    # This should only be reached if all retries fail
    return None, "All retry attempts failed", timestamp

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
    if not api_key:
        api_key = get_openai_api_key()
        if not api_key:
            st.error("❌ OpenAI API key is missing. Please check your environment variables.")
            return None, None, None
    
    timestamp = datetime.now()
    st.info(f"🔹 Calling Tier-B (OpenAI) with model {model_name}...")
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Format message for the API
    messages = [
        {"role": "system", "content": "You are a meticulous taxonomy auditor specializing in structured taxonomies."},
        {"role": "user", "content": prompt}
    ]
    
    # API call with retry logic
    for retry_count in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                top_p=1
            )
            
            # If we get here, the API call succeeded
            timestamp = datetime.now()
            
            if hasattr(response, 'choices') and len(response.choices) > 0:
                raw_response = response.choices[0].message.content
                
                if raw_response:
                    return raw_response, raw_response, timestamp
                else:
                    st.warning(f"⚠️ Model {model_name} returned empty content")
                    return None, None, timestamp
            else:
                st.error(f"❌ Error: API response missing choices")
                return None, None, timestamp
                
        except OpenAI_RateLimitError as e:
            if retry_count < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[retry_count]
                st.warning(f"⚠️ Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error(f"❌ Rate limit exceeded after {MAX_RETRIES} retries.")
                return None, str(e), timestamp
                
        except OpenAI_APIError as e:
            st.error(f"❌ API Error: {str(e)}")
            return None, str(e), timestamp
            
        except OpenAI_AuthError as e:
            st.error(f"❌ Authentication Error: Please check your API key.")
            return None, str(e), timestamp
            
        except OpenAI_ConnError as e:
            if retry_count < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[retry_count]
                st.warning(f"⚠️ Connection error. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error(f"❌ Connection error after {MAX_RETRIES} retries.")
                return None, str(e), timestamp
                
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None, str(e), timestamp
            
    # This should only be reached if all retries fail
    return None, "All retry attempts failed", timestamp