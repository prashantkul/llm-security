# llm-security
This repository contains the code to test various security issues with LLM prompts. It has detectors to detect these issues. It shows how to invoke these detectors independantly and via LangChain.

Following detectors are available:
- Google Cloud Armor product for content safety and prompt injection detection
- Prompt injection using '<aprotectai/deberta-v3-base-prompt-injection-v2'
- Various other issues like hate speech, profanity, harassement, firearms and weapons, public safety etc. 

The detectors and LLMs are invoked via Langchain. LangSmith is used for visualization and troubleshooting. 

## Features
- Google Cloud Model Armor for a comprehensive prompt attacks. (Feature in Private preview, not yet GA)
- Huggingface DeBerta model for prompt injection detection.
- LangChain and LangSmith integration to show the flow of the verifications

## Models
- Huggingface
- Google Gemini
- OpenAI GPT
- Anthropic Claude

# Setup
## Requirements
- Need a Google Cloud project that is enabled for Model Armor
- Store API keys as a secrets in Google Cloud Secrets Manager

## Service account
- Create a service account in this Google Cloud project
- Assign following roles at project level
  - 'Secret Manager Secret Accessor' (this role can be assigned at secret level as well for tigher security control)
  - 'Model Armor Admin' (Your project needs to be allow listed to use this preview feature, for now)

## Secrets
Create following secrets in the Google Cloud project:
- anthropic_api_key
- google_api_key
- huggingface_api_key
- langsmith_api_key
- openai_api_key

## Input
- Provide a local.env file containing following key-value pairs
  - SA_KEY_PATH=\<path-to-key-file\>
  - SA_KEY_NAME=sa-key.json
  - GOOGLE_CLOUD_PROJECT=\<project-id\>
  - MA_TEMPLATE_NAME=\<template-name\>
  - REGION=us-central1
  - PROMPT_CSV=\<path\>/prompts_and_labels.csv
  - LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
  - LANGCHAIN_TRACING_V2=true
  - LANGCHAIN_PROJECT=llm-security
  - TOKENIZERS_PARALLELISM="false"

## Output
- Provides a command line output for various prompt detection
- Generates a CSV file with summary.

## Critical libraries
- Python==3.11.5
- transformers==4.40.0
- tensorflow==2.15.0
- keras==2.15.0
- torch==2.4.1
- langsmith==0.1.117
- langchain==0.2.16

