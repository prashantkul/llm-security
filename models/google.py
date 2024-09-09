import os
import google.generativeai as genai
from typing import List, Dict, Optional


class GoogleGeminiWrapper:
    def __init__(self, model: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.chat = self.model.start_chat(history=[])

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        if role == "human":
            self.chat.history.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            self.chat.history.append({"role": "model", "parts": [content]})
        else:
            raise ValueError("Role must be either 'human' or 'assistant'")

    def set_system_message(self, content: str):
        """Set or update the system message."""
        # Gemini doesn't have a distinct system message, so we'll add it as a human message
        if (
            self.chat.history
            and self.chat.history[0]["role"] == "user"
            and self.chat.history[0]["parts"][0].startswith("System:")
        ):
            self.chat.history[0] = {"role": "user", "parts": [f"System: {content}"]}
        else:
            self.chat.history.insert(
                0, {"role": "user", "parts": [f"System: {content}"]}
            )

    def generate_response(
        self,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 1,
        max_output_tokens: int = 2048,
    ) -> str:
        """Generate a response from Gemini."""
        try:
            response = self.chat.send_message(
                "",  # Empty string because we're using the chat history
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=max_output_tokens,
                ),
            )
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def chat_with_gemini(self, user_input: str) -> str:
        """Add a user message and generate a response."""
        self.add_message("human", user_input)
        response = self.generate_response()
        self.add_message("assistant", response)
        return response

    def clear_conversation(self):
        """Clear the conversation history, keeping the system message if it exists."""
        system_message = None
        if (
            self.chat.history
            and self.chat.history[0]["role"] == "user"
            and self.chat.history[0]["parts"][0].startswith("System:")
        ):
            system_message = self.chat.history[0]
        self.chat = self.model.start_chat(history=[])
        if system_message:
            self.chat.history.append(system_message)
