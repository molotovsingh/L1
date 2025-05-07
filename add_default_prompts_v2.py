"""
Add improved default prompts to the database (Default Prompt 2)

This script creates standardized default prompts that address inconsistencies
between API providers and tiers. The new prompts use consistent parameters and
clearer instructions to avoid the "breach of fmax label" issues.
"""

import sys
import os
import datetime
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import database models
try:
    import db_models
except ImportError:
    logging.error("Could not import db_models. Make sure the module is available.")
    sys.exit(1)

# New improved default prompts with consistent parameters
DEFAULT_PROMPTS_V2 = [
    # OpenAI Tier-A Prompt (Improved)
    {
        "name": "Default OpenAI Tier-A Prompt V2",
        "tier": "A",
        "api_provider": "OpenAI",
        "content": """You are a domain taxonomy generator specializing in discrete events.

Domain: {{domain}}

TASK:
Generate a list of {{min_labels}}–{{max_labels}} distinct, top-level categories representing *specific types of events* or *discrete occurrences* within the '{{domain}}' domain. Think incidents, launches, breakthroughs, failures, breaches, discoveries, releases, regulatory actions, etc.

Rules for Labels:
1. Format: TitleCase, 1–4 words with hyphens between words (e.g., Data-Breach, System-Outage).
2. Event-Driven: Each label MUST represent a discrete event, incident, change, or occurrence. NOT ongoing states or themes.
3. Specificity: Prefer specific event types over broad categories.
4. Exclusion: Avoid terms like {{deny_list}}.
5. Output Format: Return ONLY a JSON array of strings. Example: ["Model-Launch", "System-Outage", "Major-Discovery"]

Generate the JSON array now.""",
        "description": "Improved OpenAI prompt for Tier-A with clearer event focus",
        "is_system": True
    },
    
    # OpenAI Tier-B Prompt (Improved)
    {
        "name": "Default OpenAI Tier-B Prompt V2",
        "tier": "B",
        "api_provider": "OpenAI",
        "content": """You are a meticulous taxonomy auditor enforcing specific principles.

Candidate Event Labels for Domain '{{domain}}':
{{candidates_json}}

Your Task:
Review the candidate labels based on the following principles and return a refined list.

Principles to Enforce:
1. Event-Driven Focus: Each label MUST represent a discrete event, incident, change, or occurrence. Reject labels describing general themes, capabilities, technologies, or ongoing states.
2. Formatting: Labels must be 1–4 words, TitleCase, with hyphens between words (e.g., "Data-Breach").
3. Deny List: Reject any label containing the terms: {{deny_list}}.
4. Consolidation & Target Count: Merge synonyms or similar event types. Aim for exactly {{min_labels}} labels (±1). This means between {{min_labels-1}} and {{min_labels+1}} labels.
5. Output Structure: Return ONLY a JSON object with these keys:
   - "approved": Array of approved labels (aim for {{min_labels}} ±1 labels)
   - "rejected": Array of rejected labels
   - "reason_rejected": Object mapping each rejected label to a reason

Example Output Format:
{
  "approved": ["Model-Launch", "System-Outage", "Regulatory-Action"],
  "rejected": ["AI Research", "Funding Round", "ProductUpdate"],
  "reason_rejected": {
    "AI Research": "Not event-driven, describes a theme.",
    "Funding Round": "Contains denied term 'Funding'.",
    "ProductUpdate": "Merged into Major-Release."
  }
}

Return only the JSON object now.""",
        "description": "Improved OpenAI prompt for Tier-B with consistent target count",
        "is_system": True
    },
    
    # Perplexity Tier-A Prompt (Improved)
    {
        "name": "Default Perplexity Tier-A Prompt V2",
        "tier": "A",
        "api_provider": "Perplexity",
        "content": """You are a domain taxonomy generator specializing in discrete events.

Domain: {{domain}}

TASK:
Generate a list of {{min_labels}}–{{max_labels}} distinct, top-level categories representing *specific types of events* or *discrete occurrences* within the '{{domain}}' domain. Think incidents, launches, breakthroughs, failures, breaches, discoveries, releases, regulatory actions, etc.

Rules for Labels:
1. Format: TitleCase, 1–4 words with hyphens between words (e.g., Data-Breach, System-Outage).
2. Event-Driven: Each label MUST represent a discrete event, incident, change, or occurrence. NOT ongoing states or themes.
3. Specificity: Prefer specific event types over broad categories.
4. Exclusion: Avoid terms like {{deny_list}}.
5. Output Format: Return ONLY a JSON array of strings. Example: ["Model-Launch", "System-Outage", "Major-Discovery"]

Generate the JSON array now.""",
        "description": "Improved Perplexity prompt for Tier-A with clearer event focus",
        "is_system": True
    },
    
    # Perplexity Tier-B Prompt (Improved for non-reasoning models)
    {
        "name": "Default Perplexity Tier-B Prompt V2",
        "tier": "B",
        "api_provider": "Perplexity",
        "content": """You are a meticulous taxonomy auditor enforcing specific principles.

Candidate Event Labels for Domain '{{domain}}':
{{candidates_json}}

Your Task:
Review the candidate labels based on the following principles and return a refined list.

Principles to Enforce:
1. Event-Driven Focus: Each label MUST represent a discrete event, incident, change, or occurrence. Reject labels describing general themes, capabilities, technologies, or ongoing states.
2. Formatting: Labels must be 1–4 words, TitleCase, with hyphens between words (e.g., "Data-Breach").
3. Deny List: Reject any label containing the terms: {{deny_list}}.
4. Consolidation & Target Count: Merge synonyms or similar event types. Aim for exactly {{min_labels}} labels (±1). This means between {{min_labels-1}} and {{min_labels+1}} labels.
5. Output Format: For non-reasoning models, return a JSON object as shown in the example below.

Example Output Format:
{
  "approved": ["Model-Launch", "System-Outage", "Regulatory-Action"],
  "rejected": ["AI Research", "Funding Round", "ProductUpdate"],
  "reason_rejected": {
    "AI Research": "Not event-driven, describes a theme.",
    "Funding Round": "Contains denied term 'Funding'.",
    "ProductUpdate": "Merged into Major-Release."
  }
}

Return only the JSON object now.""",
        "description": "Improved Perplexity prompt for Tier-B with consistent target count",
        "is_system": True
    },
    
    # Perplexity Tier-B Prompt (Improved for reasoning models)
    {
        "name": "Default Perplexity Tier-B Reasoning Prompt V2",
        "tier": "B",
        "api_provider": "Perplexity-Reasoning",
        "content": """You are a meticulous taxonomy auditor enforcing specific principles.

Candidate Event Labels for Domain '{{domain}}':
{{candidates_json}}

Your Task:
Review the candidate labels based on the following principles and return a refined list.

Principles to Enforce:
1. Event-Driven Focus: Each label MUST represent a discrete event, incident, change, or occurrence. Reject labels describing general themes, capabilities, technologies, or ongoing states.
2. Formatting: Labels must be 1–4 words, TitleCase, with hyphens between words (e.g., "Data-Breach").
3. Deny List: Reject any label containing the terms: {{deny_list}}.
4. Consolidation & Target Count: Merge synonyms or similar event types. Aim for exactly {{min_labels}} labels (±1). This means between {{min_labels-1}} and {{min_labels+1}} labels.

5. Output Format: For reasoning models, use this EXACT format:
<thinking>
Your detailed analysis process here...
</thinking>

APPROVED LABELS:
[List each approved label, one per line]

REJECTED LABELS:
[List each rejected label, one per line]

REJECTION REASONS:
[For each rejected label: "Label: Reason for rejection"]

The <thinking> tags help me understand your reasoning process, but I'll extract only the final lists after those tags.""",
        "description": "Improved Perplexity prompt for reasoning models with explicit thinking section",
        "is_system": True
    }
]

def add_default_prompts_v2() -> None:
    """Add the improved default prompts to the database."""
    timestamp = datetime.datetime.now()
    
    for prompt in DEFAULT_PROMPTS_V2:
        prompt_data = {
            "name": prompt["name"],
            "tier": prompt["tier"],
            "api_provider": prompt["api_provider"],
            "content": prompt["content"],
            "description": prompt["description"],
            "is_system": prompt["is_system"],
            "timestamp": timestamp
        }
        
        try:
            prompt_id = db_models.create_custom_prompt(
                name=prompt["name"],
                tier=prompt["tier"],
                api_provider=prompt["api_provider"],
                content=prompt["content"],
                description=prompt["description"],
                is_system=prompt["is_system"]
            )
            logging.info(f"Added prompt '{prompt['name']}' with ID: {prompt_id}")
        except Exception as e:
            logging.error(f"Failed to add prompt '{prompt['name']}': {e}")
    
    logging.info("Completed adding improved default prompts (V2)")

if __name__ == "__main__":
    # Add the improved default prompts
    add_default_prompts_v2()
    print("Added improved default prompts (V2) to the database")