"""
Simple OpenAI API Test Script

This script tests the OpenAI API with a basic request to diagnose rate limit issues.
"""

import os
import time
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("Missing OPENAI_API_KEY environment variable")
    logger.info("Checking for key in secrets...")
    # Let user know if API key is missing
    print("OPENAI_API_KEY not found in environment variables.")
    print("Please make sure it's set in the Replit Secrets tab.")
    exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def test_openai_api():
    """Make a simple request to the OpenAI API and log the response/errors"""
    
    logger.info("Starting OpenAI API test")
    
    prompt = "Hello, please respond with a short greeting."
    
    try:
        logger.info("Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using cheaper model to avoid rate limits
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )
        
        content = response.choices[0].message.content
        logger.info(f"Received response: {content}")
        
        # Get rate limit headers if available
        logger.info("Checking rate limit headers...")
        logger.info(f"Model used: {response.model}")
        logger.info(f"Response metadata: {response.usage}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    # Test once
    success = test_openai_api()
    
    if success:
        logger.info("First test successful. Testing rate limits with multiple requests...")
        
        # Test rate limits with a few consecutive requests
        for i in range(3):
            logger.info(f"Test request {i+1}/3...")
            test_openai_api()
            time.sleep(2)  # Small delay between requests
            
        logger.info("API test complete")
    else:
        logger.error("Initial test failed. Check API key or network connection.")