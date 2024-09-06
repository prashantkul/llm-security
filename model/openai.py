import openai
from typing import List, Dict, Optional


class OpenAIGPTWrapper:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def set_system_message(self, content: str):
        """Set or update the system message."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = {"role": "system", "content": content}
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def generate_response(
        self,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> str:
        """Generate a response from the model."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            assistant_message = response["choices"][0]["message"]["content"]
            self.add_message("assistant", assistant_message)
            return assistant_message
        except Exception as e:
            return f"Error: {str(e)}"

    def chat(self, user_input: str) -> str:
        """Add a user message and generate a response."""
        self.add_message("user", user_input)
        return self.generate_response()

    def clear_conversation(self):
        """Clear the conversation history, keeping the system message if it exists."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
