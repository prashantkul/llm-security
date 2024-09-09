import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class HuggingFaceWrapper:

    def __init__(
        self,
        model_name="bigscience/bloom-1b7",
        torch_dtype=torch.bfloat16,
    ):
        print(f"Initializing HuggingFaceWrapper with model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"  # Set padding to left
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token
        print("Model and tokenizer initialized successfully")
        self.messages = []

    def set_system_message(self, content):
        self.messages = [{"role": "system", "content": content}]

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def generate_response(
        self,
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    ):
        print("Generating response from HF model")
        print("Prompt: ", prompt)

        try:
            input_text = f"Question: {prompt}\nAnswer:"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            print(f"Input shape: {input_ids.shape}")

            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
            )
            print(f"Output shape: {output.shape}")

            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Full generated text: {generated_text}")

            response = generated_text.split("Answer:")[-1].strip()
            print(f"Generated response: {response}")
            return response

        except Exception as e:
            error_message = f"Error in generate_response: {str(e)}"
            print(error_message)
            return error_message

    def chat(self, user_input, validate_func=None):
        self.add_user_message(user_input)
        response = self.generate_response()

        if validate_func:
            response = validate_func(response)

        print(response)
        return response
