import json
import re
from typing import Dict, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


RELATION_LIST_RESPONSE_FORMAT: Dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "relation_list",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "relation_list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "head": {"type": "string"},
                            "relation": {"type": "string"},
                            "tail": {"type": "string"},
                            "evidence": {"type": "string"},
                            "skill": {"type": "string"},
                        },
                        "required": ["head", "relation", "tail", "evidence", "skill"],
                    },
                },
            },
            "required": ["relation_list"],
        },
    },
}

ROUTER_RESPONSE_FORMAT: Dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "skill_router",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "primary_skill": {"type": "string"},
                "aux_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reason": {"type": "string"},
            },
            "required": ["primary_skill", "aux_skills", "reason"],
        },
    },
}


class LocalQwenGenerator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self.model is not None and self.tokenizer is not None:
            return
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            trust_remote_code=True,
        )

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        self.load()
        assert self.tokenizer is not None and self.model is not None
        messages = [
            {"role": "system", "content": "/no_think 你是一个严格遵循输出格式的军事关系抽取系统。"},
            {"role": "user", "content": "/no_think\n请只输出一个合法 JSON 对象，第一字符必须是 {，最后字符必须是 }。\n" + prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def build_backend(
    backend: str,
    api_key: str,
    model_path: str,
) -> Tuple[Optional[OpenAI], Optional[LocalQwenGenerator]]:
    client = None
    local_generator = None
    if backend == "openai":
        if OpenAI is None:
            raise ImportError("openai module is required for openai backend")
        import httpx
        client = OpenAI(
            api_key=api_key,
            http_client=httpx.Client(trust_env=False, timeout=120.0),
        )
    elif backend == "api":
        if OpenAI is None:
            raise ImportError("openai module is required for api backend")
        import httpx
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            http_client=httpx.Client(trust_env=False, timeout=120.0),
        )
    elif backend == "qwen_api":
        if OpenAI is None:
            raise ImportError("openai module is required for qwen_api backend")
        import httpx
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=httpx.Client(trust_env=False, timeout=120.0),
        )
    elif backend == "local_qwen3":
        local_generator = LocalQwenGenerator(model_path)
    elif backend == "vllm":
        # vLLM is called through the requests fallback so it does not require
        # the OpenAI SDK in the backend runtime.
        pass
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return client, local_generator


def generate_text(
    prompt: str,
    backend: str,
    api_client: Optional[OpenAI],
    local_generator: Optional[LocalQwenGenerator],
    model: str,
    max_tokens: int,
    response_format: Optional[Dict] = None,
) -> str:
    if backend in {"openai", "api", "qwen_api"}:
        assert api_client is not None
        request_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0 if max_tokens <= 600 else 0.1,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if backend == "qwen_api":
            request_kwargs["extra_body"] = {"enable_thinking": False}
        if response_format is not None and backend in {"openai", "qwen_api"}:
            request_kwargs["response_format"] = response_format
        response = api_client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content
    assert local_generator is not None
    return local_generator.generate(prompt, max_new_tokens=max_tokens)


def generate_text_with_requests(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int,
    backend: str = "qwen_api",
    response_format: Optional[Dict] = None,
    base_url: Optional[str] = None,
    timeout: int = 120,
    enable_thinking: bool = False,
) -> str:
    import requests

    if base_url:
        resolved_base_url = base_url.rstrip("/")
    elif backend == "qwen_api":
        resolved_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif backend == "api":
        resolved_base_url = "https://api.deepseek.com"
    elif backend == "openai":
        resolved_base_url = "https://api.openai.com"
    elif backend == "vllm":
        raise ValueError("base_url is required for vllm backend")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0 if max_tokens <= 600 else 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if backend == "qwen_api":
        data["enable_thinking"] = False
    if backend == "vllm":
        data["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if response_format is not None and backend in {"openai", "qwen_api"}:
        data["response_format"] = response_format

    response = requests.post(
        f"{resolved_base_url}/chat/completions",
        headers=headers,
        json=data,
        timeout=timeout,
    )
    if response.status_code != 200:
        print(f"API Error: {response.status_code}")
        print(f"Response: {response.text}")
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]
