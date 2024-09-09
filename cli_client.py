import os
from typing import List, Dict
from models.anthropic import AnthropicClaudeWrapper
from models.google import GoogleGeminiWrapper
from models.openai import OpenAIGPTWrapper
from models.huggingface import HuggingFaceWrapper
from util.secrets import GoogleSecretManagerAPIKey
from typing import List, Dict, Optional
import torch
from dotenv import load_dotenv, find_dotenv


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Response generation timed out")


class SecureUnifiedAIModelTestClient:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.key_manager = GoogleSecretManagerAPIKey(project_id)
        self.models = {}
        self._initialize_models()
        self.injection_detector = DebertaPromptInjectionDetector()


    def _initialize_models(self):
        """Initialize all AI model wrappers using securely retrieved API keys."""
        api_keys = {
            "anthropic": self.key_manager.get_key("anthropic"),
            "google": self.key_manager.get_key("google"),
            "openai": self.key_manager.get_key("openai"),
            "huggingface": self.key_manager.get_key("huggingface"),
        }

        self.models["openai"] = OpenAIGPTWrapper()
        self.models["huggingface"] = HuggingFaceWrapper()

        # if api_keys["anthropic"]:
        #     self.models["anthropic"] = AnthropicClaudeWrapper(api_keys["anthropic"])
        # if api_keys["google"]:
        #     self.models["google"] = GoogleGeminiWrapper(api_keys["google"])
        # if api_keys["openai"]:
        #     self.models["openai"] = OpenAICodexWrapper(api_keys["openai"])
        # if api_keys["huggingface"]:
        #     self.models["huggingface"] = HuggingFaceWrapper(api_keys["huggingface"])

    def batch_test(self, model_names: List[str], prompts: List[str]) -> List[dict]:
        """Batch test multiple prompts on multiple or all models."""
        results = []
        model_names = model_names or list(self.models.keys())
        for prompt in prompts:
            for model_name in model_names:
                results.append(self.test_model(model_name, prompt))
        return results

    def test_model(self, model_name: str, prompt: str) -> str:
        """Test a specific model with the given prompt."""
        print(f"Testing {model_name}")
        if model_name not in self.models:
            return f"Error: Model {model_name} not initialized."
        try:
            return self.models[model_name].generate_response(prompt)
        except Exception as e:
            return f"Error with {model_name}: {str(e)}"

    def test_all_models(self, prompt: str) -> Dict[str, str]:
        """Test all initialized models with the given prompt."""
        results = {}
        for model_name in self.models:
            results[model_name] = self.test_model(model_name, prompt)
        return results

    def compare_models(self, prompts: List[str]):
        """Compare responses from all models for multiple prompts."""
        for i, prompt in enumerate(prompts, 1):
            print(f"\nTesting Prompt {i}: {prompt}")
            responses = self.test_all_models(prompt)
            for model, response in responses.items():
                print(f"{model.capitalize()} response: {response}")

    def run_interactive_session(self):
        """Run an interactive session allowing user to input prompts."""
        print("\nStarting interactive session. Type 'exit' to end the session.")
        while True:
            prompt = input("\nEnter your prompt: ").strip()
            if prompt.lower() == "exit":
                break
            responses = self.test_all_models(prompt)
            for model, response in responses.items():
                print(f"{model.capitalize()} response: {response}")
    
    def test_model_with_injection_detection(self, model_name: str, prompt: str) -> Dict[str, Union[str, Dict]]:
        """Test a specific model with the given prompt and perform injection detection."""
        injection_result = self.injection_detector.test_prompt(prompt)
        
        if injection_result["is_injection"]:
            return {
                "model_response": f"Potential prompt injection detected. Status: {injection_result['status']}",
                "injection_detection": injection_result
            }
        
        model_response = self.test_model(model_name, prompt)
        return {
            "model_response": model_response,
            "injection_detection": injection_result
        }

    def batch_test_with_injection_detection(self, model_names: List[str], prompts: List[str]) -> List[Dict]:
        """Batch test multiple prompts on multiple or all models with injection detection."""
        results = []
        model_names = model_names or list(self.models.keys())
        for prompt in prompts:
            for model_name in model_names:
                results.append(self.test_model_with_injection_detection(model_name, prompt))
        return results

    def compare_models_with_injection_detection(self, prompts: List[str]):
        """Compare responses from all models for multiple prompts with injection detection."""
        for i, prompt in enumerate(prompts, 1):
            print(f"\nTesting Prompt {i}: {prompt}")
            injection_result = self.injection_detector.test_prompt(prompt)
            print(f"Injection detection result: {injection_result['status']}")
            
            if not injection_result["is_injection"]:
                responses = self.test_all_models(prompt)
                for model, response in responses.items():
                    print(f"{model.capitalize()} response: {response}")
            else:
                print(f"Prompt injection detected. Status: {injection_result['status']}")


if __name__ == "__main__":
    torch.set_default_device("cpu")
    import logging

    # logging.basicConfig(level=logging.DEBUG)  # uncomment to debug the environment
    print("Loading .env file")
    _ = load_dotenv('local.env')
    # set Google Cloud project
    project_id = os.environ["GCP_PROJECT_ID"]
    if not project_id:
        raise ValueError("Please set the GOOGLE_CLOUD_PROJECT environment variable")

    test_client = SecureUnifiedAIModelTestClient(project_id)

    # Test with predefined prompts
    test_prompts = [
        "Explain the concept of artificial intelligence in one sentence.",
        "What are the three laws of robotics?",
        "Write a haiku about machine learning.",
    ]
    print("About to run tests...")

    # test_client.compare_models(test_prompts)

    # # Run interactive session
    # test_client.run_interactive_session()
    output = test_client.batch_test(["huggingface"], test_prompts)
    print(f"Response: {output}")
