from openai import OpenAI
import os
import model_mapper  # Import our custom model mapper

# 1.  Configure the client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),     # set this in Replit Secrets
    organization=os.getenv("OPENAI_ORG_ID")  # optional, if you have several orgs
)

# Map the shorthand model name to the full name
model_shorthand = "o3"
model_fullname = model_mapper.map_model_name(model_shorthand)

print(f"Using model: {model_shorthand} → mapped to: {model_fullname}")

# 2.  Make a single chat completion
response = client.chat.completions.create(
    model=model_fullname,  # Use the mapped model name
    messages=[
        {"role": "user", "content": "In one sentence, explain what causes a rainbow."}
    ],
    max_tokens=50,
    stream=False            # flip to True if you want chunk-by-chunk streaming
)

# 3.  Print the model's answer
print(response.choices[0].message.content.strip())