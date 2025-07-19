import torch


class TokenizerChatTemplate:
    def __init__(self, tokenizer, add_special_tokens, system_prompt, empty_template=False):
        self.empty_template = empty_template
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

        self.system_prompt = system_prompt

    def apply(self, messages, add_generation_prompt=True):
        if self.system_prompt is not None:
            messages = [dict(role="system", content=self.system_prompt)] + messages

        extra_kwargs = {}

        if "Qwen3" in self.tokenizer.name_or_path:
            extra_kwargs["enable_thinking"] = False

        if not self.empty_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                return_dict=False,
                tokenize=False,
                **extra_kwargs
            )
        else:
            prompt = "\n".join(message['content'] for message in messages)

        inputs = self.tokenizer(
            prompt,
            padding=False,
            truncation=False,
            max_length=None,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )
        
        return inputs


def get_conv_template(model_name, tokenizer):
    LLAMA_MODELS = (
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "llamafactory/tiny-random-Llama-3",
    )

    QWEN_MODELS = (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
    )

    if model_name in LLAMA_MODELS:
        return TokenizerChatTemplate(tokenizer, add_special_tokens=False, system_prompt="A chat between a curious human and an artificial intelligence assistant. " \
            "The assistant gives helpful, detailed, and polite answers to the human's questions.")  # special tokens already included in template
    elif model_name in QWEN_MODELS:
        return TokenizerChatTemplate(tokenizer, add_special_tokens=False, system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
    else:
        raise NotImplementedError(f"Model name '{model_name}' is not supported.")


def apply_chat_template(model, tokenizer, input):
    if isinstance(input, list):
        assert len(input) > 0, "Input list is empty."
        assert isinstance(input[0], dict)
        messages = input
    else:
        assert isinstance(input, str)
        messages = [
            {"role": "user", "content": input},
        ]
    for message in messages:
        assert isinstance(message, dict), f"Expected message to be a dictionary, got {type(message)}"
        assert "content" in message
        assert message["role"] in ["user", "system", "assistant"]
        assert message["role"] in ["user", "system", "assistant"]
    add_generation_prompt = len(messages) > 0 and messages[-1]["role"] == "user"
    add_generation_prompt = messages[-1]["role"] == "user"
    return get_conv_template(model.config.name_or_path, tokenizer).apply(messages, add_generation_prompt=add_generation_prompt)


def tokenize_input(model, tokenizer, data, input_txt):
    input_full = input_txt

    if data.use_chat_template:
        inputs = apply_chat_template(model, tokenizer, input_txt)
    else:
        input_full = input_txt
        inputs = tokenizer(input_full, add_special_tokens=data.add_special_tokens, return_tensors="pt")

    return inputs


class InputProcessor:
    def __init__(self, model, tokenizer, data):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data

    def encode(self, input_txt, device=None):
        encoded_inputs = tokenize_input(self.model, self.tokenizer, self.data, input_txt)
        
        if device is not None:
            encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

        return encoded_inputs
    
    def decode(self, input_ids):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()
        return self.tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)