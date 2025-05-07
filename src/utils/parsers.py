"""
Parser utilities for processing API responses in various formats.

This module provides flexible parsing functions for extracting structured
data from different response formats, including JSON, plain text lists,
and unstructured text with embedded data.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional

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


def extract_structured_data(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract structured data from text that may contain a JSON object.
    
    Args:
        text: Text that may contain a JSON object
        
    Returns:
        Dict or None: Structured data extracted from text
    """
    if not text:
        return None
        
    # Try direct JSON parsing first
    try:
        data = json.loads(text.strip())
        return data
    except json.JSONDecodeError:
        pass
    
    # Try to find a JSON object in the text
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return data
    except Exception:
        pass
    
    return None