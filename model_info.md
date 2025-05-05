# OpenAI Model Guide for Taxonomy Generation

## Model Families

### 1. GPT-4 Series (Multimodal)
- **Models**: gpt-4o, gpt-4-turbo, gpt-4
- **Strengths**: Complete feature set, image processing, audio capabilities
- **Best for**: General-purpose taxonomy generation, handling all prompt types
- **Recommended**: `gpt-4o` (most capable) or `gpt-3.5-turbo` (most economical)

### 2. o-Series (Reasoning)
- **Models**: o1, o3, o4-mini
- **Strengths**: Superior reasoning, logic, and structured thinking
- **Limitations**: 
  - Text-only (no image/audio capabilities)
  - May return empty responses depending on account permissions
  - Requires special parameters (max_completion_tokens)
- **Usage Notes**: These models require special org-level access for full functionality

### 3. GPT-3.5 Series
- **Models**: gpt-3.5-turbo
- **Strengths**: High rate limits, fast responses, more economical
- **Best for**: Quick iterations and testing

## Usage Tips

1. **For most taxonomy generation**: Use `gpt-4o` or `gpt-3.5-turbo`
2. **If experiencing empty responses**: Switch from o-series to GPT-4 series models
3. **For deep reasoning tasks**: If you have full access to o-series, they may provide better structured output

## Parameter Differences

| Model Family | Key Parameters | Notes |
|--------------|----------------|-------|
| GPT-4 Series | max_tokens, temperature | Full support for all parameters |
| o-Series | max_completion_tokens | Temperature fixed at 1.0 |

## Rate Limits & Costs (As of May 2025)

| Model | TPM / RPM | Cost (per 1K tokens) |
|-------|-----------|----------------------|
| o1 | 30k TPM · 500 RPM | $0.06 in / $0.30 out |
| o3 | 30k TPM · 500 RPM | $0.12 in / $0.60 out |
| gpt-4o | 30k TPM · 500 RPM | $0.50 in / $1.50 out |
| gpt-4o-mini | 200k TPM · 500 RPM | $0.15 in / $0.60 out |
| gpt-3.5-turbo | 300k TPM · 3500 RPM | $0.005 in / $0.015 out |

*Numbers from April 2025 pricing; check dashboard for current rates*