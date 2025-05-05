"""
Simple OpenAI API Test with GPT-3.5-turbo

This script tests if our API key works with a standard model
"""

from openai import OpenAI
import os
import json

# Get API key from environment
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

def test_standard_model():
    """Test with a standard model (gpt-3.5-turbo)"""
    print("Testing with GPT-3.5-turbo:")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello, what's 2+2?"}
            ],
            max_tokens=10
        )
        print(f"Response content: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_o1_model():
    """Test with o1 model"""
    print("\nTesting with o1 model:")
    try:
        response = client.chat.completions.create(
            model="o1",
            messages=[
                {"role": "user", "content": "Hello, what's 2+2?"}
            ],
            max_completion_tokens=10
        )
        print(f"Response type: {type(response)}")
        print(f"Choices count: {len(response.choices)}")
        
        if len(response.choices) > 0:
            message = response.choices[0].message
            print(f"Message role: {message.role}")
            print(f"Content: '{message.content}'")
        else:
            print("No choices returned in the response")
        
        print(f"\nFull response:")
        # Print all fields of response excluding large objects
        for key, value in response.model_dump().items():
            if key != "choices" and key != "usage":
                print(f"  {key}: {value}")
        print(f"  usage: {response.usage}")
        
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice = response.choices[0]
            choice_dict = choice.model_dump()
            print(f"  choices[0]:")
            for key, value in choice_dict.items():
                if key != "message":
                    print(f"    {key}: {value}")
            
            if hasattr(choice, "message"):
                message_dict = choice.message.model_dump()
                print(f"    message:")
                for key, value in message_dict.items():
                    print(f"      {key}: {value}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Test standard model first
    if test_standard_model():
        # If that works, test o1 model
        test_o1_model()