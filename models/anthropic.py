import os
import anthropic
from typing import List, Dict, Optional


class AnthropicClaudeWrapper:
    def __init__(self, model: str = "claude-3-opus-20240229"):

        if not api_key:
            raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")
        self.client = anthropic.Anthropic(api_key=api_key)
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
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """Generate a response from Claude."""
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=self.messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            assistant_message = response.content[0].text
            self.add_message("assistant", assistant_message)
            return assistant_message
        except Exception as e:
            return f"Error: {str(e)}"

    def chat(self, user_input: str) -> str:
        """Add a user message and generate a response."""
        self.add_message("human", user_input)
        return self.generate_response()

    def clear_conversation(self):
        """Clear the conversation history, keeping the system message if it exists."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
