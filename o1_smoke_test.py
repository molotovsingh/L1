from openai import OpenAI
import os
import model_mapper  # Import our custom model mapper

# 1.  Configure the client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),     # set this in Replit Secrets
    organization=os.getenv("OPENAI_ORG_ID")  # optional, if you have several orgs
)

# Map the shorthand model name to the full name
model_shorthand = "o1"
model_fullname = model_mapper.map_model_name(model_shorthand)

print(f"Using model: {model_shorthand} → mapped to: {model_fullname}")

# 2.  Make a single chat completion
response = client.chat.completions.create(
    model=model_fullname,  # Use the mapped model name
    messages=[
        {"role": "user", "content": "In one sentence, explain why the sky appears blue."}
    ],
    max_completion_tokens=50,  # o-series models use max_completion_tokens instead of max_tokens
    stream=False            # flip to True if you want chunk-by-chunk streaming
)

# 3.  Print the response details
print("\nResponse details:")
print(f"Response type: {type(response)}")
print(f"Choices count: {len(response.choices)}")

if len(response.choices) > 0:
    message = response.choices[0].message
    print(f"Message role: {message.role}")
    print(f"Content type: {type(message.content)}")
    print(f"Content empty?: {message.content == ''}")
    print(f"Content: {message.content.strip() if message.content else 'EMPTY'}")
else:
    print("No choices returned in the response")