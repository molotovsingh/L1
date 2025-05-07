"""
Update token limits for Perplexity API calls

This script increases the max_tokens parameter in the Perplexity API calls, 
particularly for Tier-B reasoning models that require more tokens to 
complete their detailed analysis without truncation.
"""

import sys
import re
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_file(filename: str) -> bool:
    """
    Update the max_tokens parameter in API calls in the specified file.
    
    Args:
        filename: Path to the file to update
        
    Returns:
        bool: True if updates were made, False otherwise
    """
    try:
        # Read the file
        with open(filename, 'r') as f:
            content = f.read()
        
        # Create backup of original file
        with open(f"{filename}.bak", 'w') as f:
            f.write(content)
        logging.info(f"Created backup of {filename} at {filename}.bak")
        
        # Replace max_tokens in Tier-B API calls
        tier_b_pattern = r'def call_perplexity_api_tier_b.*?max_tokens=2048,'
        tier_b_replacement = lambda m: m.group(0).replace(
            'max_tokens=2048,', 
            'max_tokens=4096,  # Increased from 2048 to handle larger reasoning responses'
        )
        content_updated = re.sub(
            tier_b_pattern, 
            tier_b_replacement, 
            content, 
            flags=re.DOTALL
        )
        
        # Also update the extract_structured_data methods with higher token limits
        extract_pattern = r'def extract_structured_data.*?max_tokens=2048,'
        extract_replacement = lambda m: m.group(0).replace(
            'max_tokens=2048,', 
            'max_tokens=4096,  # Increased from 2048 to handle larger data extraction'
        )
        content_updated = re.sub(
            extract_pattern, 
            extract_replacement, 
            content_updated, 
            flags=re.DOTALL
        )
        
        # Check if changes were made
        if content_updated == content:
            logging.warning(f"No changes were needed in {filename}")
            return False
        
        # Write the updated content
        with open(filename, 'w') as f:
            f.write(content_updated)
        
        logging.info(f"Updated {filename} with increased token limits")
        return True
        
    except Exception as e:
        logging.error(f"Error updating {filename}: {e}")
        return False

def main():
    """Main function to update Perplexity API token limits."""
    files_to_update = ['call_perplexity_api.py']
    
    success_count = 0
    for file in files_to_update:
        if update_file(file):
            success_count += 1
    
    logging.info(f"Updated {success_count}/{len(files_to_update)} files with increased token limits")
    
    if success_count == len(files_to_update):
        print("\nToken limits have been successfully increased for Perplexity API calls.")
        print("This should help prevent truncation in Tier-B reasoning model responses.")
    else:
        print("\nSome files could not be updated. Check the logs for details.")

if __name__ == "__main__":
    main()