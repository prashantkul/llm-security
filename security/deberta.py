from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Union


class DebertaPromptInjectionDetector:
    def __init__(
        self, model_name: str = "protectai/deberta-v3-base-prompt-injection-v2"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )

    def detect_injection(self, input_text: str) -> Dict[str, Union[bool, float]]:
        """
        Detect if the input text contains a potential prompt injection.

        Args:
            input_text (str): The text to check for prompt injection.

        Returns:
            Dict[str, Union[bool, float]]: A dictionary containing the detection result and confidence score.
        """
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        injection_prob = probabilities[0][1].item()

        is_injection = injection_prob > 0.5

        return {"is_injection": is_injection, "injection_probability": injection_prob}

    def test_prompt(
        self, prompt: str, threshold: float = 0.5
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Test a prompt for potential injection and provide a detailed result.

        Args:
            prompt (str): The prompt to test.
            threshold (float): The probability threshold for classifying as an injection (default: 0.5).

        Returns:
            Dict[str, Union[bool, float, str]]: A dictionary containing the test results.
        """
        result = self.detect_injection(prompt)

        injection_probability = result["injection_probability"]
        is_injection = injection_probability > threshold

        if is_injection:
            status = "HIGH RISK" if injection_probability > 0.8 else "MODERATE RISK"
        else:
            status = "LOW RISK" if injection_probability > 0.2 else "SAFE"

        return {
            "prompt": prompt,
            "is_injection": is_injection,
            "injection_probability": injection_probability,
            "status": status,
            "threshold_used": threshold,
        }
