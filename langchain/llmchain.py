from langsmith import Client
import langchain
from langsmith.utils import LangSmithConflictError

from typing import Any, Dict
import uuid
from datetime import datetime
import time
import logging
from dotenv import load_dotenv
import os, json

from util.secrets import GoogleSecretManagerAPIKey
from util.googleauth import GoogleAuthManager
from models.openai import OpenAIGPTWrapper
from models.anthropic import AnthropicClaudeWrapper
from models.google import GoogleGeminiWrapper
from models.huggingface import HuggingFaceWrapper
from security.deberta import DebertaPromptInjectionDetector
import warnings
import logging

# Set up logging to show debug output
logging.basicConfig(level=logging.DEBUG)

warnings.filterwarnings("ignore", category=FutureWarning)
langchain.verbose = True
langchain.debug = True
langchain.llm_cache = False

# Load environment variables
load_dotenv("local.env")

# Set up Google Secret Manager
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
if not project_id:
    raise ValueError("Please set the GOOGLE_CLOUD_PROJECT environment variable")

sa_key_path = os.path.join(os.getenv("SA_KEY_PATH"), os.getenv("SA_KEY_NAME"))
auth_manager = GoogleAuthManager(sa_key_path)
secret_manager = GoogleSecretManagerAPIKey(project_id, auth_manager)
openai_api_key = secret_manager.get_key("openai")

# Set up LangSmith
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
langsmith_api_key = secret_manager.get_key("langsmith")
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key


class LangChainFlow:
    def __init__(self, model_name: str):
        self.prompt_injection_detector = PromptInjectionDetector()
        self.client = Client()
        self.model = self._initialize_model(model_name)

    def _initialize_model(self, model_name: str):
        if model_name == "openai":
            return OpenAIGPTWrapper(api_key=openai_api_key)
        elif model_name == "anthropic":
            return AnthropicClaudeWrapper()
        elif model_name == "google":
            return GoogleGeminiWrapper()
        elif model_name == "huggingface":
            return HuggingFaceWrapper()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def detect_injection(self, prompt: str) -> bool:
        return self.prompt_injection_detector.detect_injection(prompt)

    def call_llm(self, prompt: str) -> str:
        return self.model.chat(prompt)

    def run_flow(self, prompt: str) -> Dict[str, Any]:
        parent_run_id = str(uuid.uuid4())  # Generate a unique parent run ID
        try:
            # Create parent run with known ID
            logging.debug(f"Creating parent run with id {parent_run_id}")
            self.client.create_run(
                project_name="llm-security",
                name="LangChainFlow Execution",
                inputs={"prompt": prompt},
                run_type="chain",
                id=parent_run_id,  # Pass the parent_run_id that we generated
            )

            # Simulate waiting for the run to fully initialize
            logging.debug("Waiting for run to fully initialize...")
            time.sleep(5)  # Add a short delay if needed

            # Step 1: Detect prompt injection
            injection_run_id = str(uuid.uuid4())  # Generate unique ID for injection run
            logging.debug(f"Creating injection run with id {injection_run_id}")
            self.client.create_run(
                name="Prompt Injection Detection",
                inputs={"prompt": prompt},
                run_type="tool",
                parent_run_id=parent_run_id,  # Associate with the parent run
                id=injection_run_id,  # Pass a unique ID for the injection run
            )

            injection_detected = self.detect_injection(prompt)

            # Update the injection run with the result
            logging.debug(f"Updating injection run with id {injection_run_id}")
            self.client.update_run(
                injection_run_id,
                outputs={"injection_detected": injection_detected},
                end_time=datetime.utcnow(),
            )

            if injection_detected:
                # If prompt injection is detected, end the parent run
                logging.debug(
                    f"Updating parent run {parent_run_id} with injection detected."
                )
                self.client.update_run(
                    parent_run_id,
                    error="Prompt injection detected",
                    outputs={"error": "Prompt injection detected"},
                    end_time=datetime.utcnow(),
                )
                return {"error": "Prompt injection detected"}

            # Step 2: Call LLM if no injection is detected
            llm_run_id = str(uuid.uuid4())  # Generate unique ID for LLM call run
            logging.debug(f"Creating LLM call run with id {llm_run_id}")
            self.client.create_run(
                name="LLM Call",
                inputs={"prompt": prompt},
                run_type="llm",
                parent_run_id=parent_run_id,  # Associate with the parent run
                id=llm_run_id,  # Pass a unique ID for the LLM run
            )

            response = self.call_llm(prompt)

            # Update the LLM run with the result
            logging.debug(f"Updating LLM run with id {llm_run_id}")
            self.client.update_run(
                llm_run_id, outputs={"response": response}, end_time=datetime.utcnow()
            )

            # Update the parent run with the final result
            logging.debug(f"Updating parent run {parent_run_id} with final response.")
            self.client.update_run(
                parent_run_id,
                outputs={"response": response},
                end_time=datetime.utcnow(),
            )

            return {"response": response}

        except Exception as e:
            logging.error(f"Error during execution: {e}")
            if parent_run_id:
                self.client.update_run(
                    parent_run_id,
                    error=str(e),
                    outputs={"error": str(e)},
                    end_time=datetime.utcnow(),
                )
            return {"error": str(e)}

        finally:
            if parent_run_id:
                try:
                    # Ensure the parent run is always finalized
                    logging.debug(f"Finalizing parent run {parent_run_id}")
                    self.client.update_run(parent_run_id, end_time=datetime.utcnow())
                except LangSmithConflictError:
                    logging.warning(
                        f"Parent run {parent_run_id} was already finalized."
                    )


class PromptInjectionDetector:
    def __init__(self):
        self.injection_detector = DebertaPromptInjectionDetector()

    def detect_injection(self, prompt: str) -> bool:
        result = self.injection_detector.test_prompt(prompt)
        return result["is_injection"]


# Usage
model_name = "openai"
prompt = "What is the capital of France?"

langchain_flow = LangChainFlow(model_name=model_name)
result = langchain_flow.run_flow(prompt)
print(result)
