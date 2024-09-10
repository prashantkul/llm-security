# llm-security
This repository contains the code to test various security issues with LLM prompts

## Features
- Google Cloud Model Armor for a comprehensive prompt attacks.
- Huggingface DeBerta model for prompt injection detection.

## Models
- Huggingface
- Google Gemini
- OpenAI GPT
- Anthropic Claude

## Requirements
- Need a Google Cloud project that is enabled for Model Armor
- Store API keys as a secrets in Google Cloud Secrets Manager

## Input
- Provide a local.env file containing following key-value pairs
  - SA_KEY_PATH=\<path-to-key-file\>
  - SA_KEY_NAME=sa-key.json
  - GOOGLE_CLOUD_PROJECT=\<project-id\>
  - MA_TEMPLATE_NAME=\<template-name\>
  - REGION=us-central1
  - PROMPT_CSV=\<path\>/prompts_and_labels.csv

## Output
- Provides a command line output for various prompt detection
- Generates a CSV file with summary.

## Critical libraries versions
Python==3.11.5
transformers==4.40.0
tensorflow==2.15.0
keras==2.15.0
torch==2.4.1

