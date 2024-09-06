import torch
from transformers import pipeline


class HuggingFaceChatWrapper:
    def __init__(
        self,
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
    ):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.messages = []

    def set_system_message(self, content):
        self.messages = [{"role": "system", "content": content}]

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def generate_response(
        self, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95
    ):
        prompt = self.pipe.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return outputs[0]["generated_text"]

    def chat(self, user_input, validate_func=None):
        self.add_user_message(user_input)
        response = self.generate_response()

        if validate_func:
            response = validate_func(response)

        print(response)
        return response
