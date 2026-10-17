import codecs
import json
import logging
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

from openai import DefaultHttpxClient, OpenAI

logger = logging.getLogger(__name__)


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
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}

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
    base_url: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 3,
) -> Tuple[Optional[OpenAI], Optional[LocalQwenGenerator]]:
    client = None
    local_generator = None
    if backend in {"openai", "api", "qwen_api"}:
        default_urls = {"openai": "https://api.openai.com/v1", "api": "https://api.deepseek.com",
                        "qwen_api": "https://dashscope.aliyuncs.com/compatible-mode/v1"}
        client = OpenAI(
            api_key=lambda: api_key,
            base_url=normalize_openai_base_url(base_url or default_urls[backend]),
            timeout=timeout, max_retries=max_retries - 1,
            http_client=DefaultHttpxClient(trust_env=False, timeout=timeout),
        )
    elif backend == "local_qwen3":
        local_generator = LocalQwenGenerator(model_path)
    elif backend == "vllm":
        # vLLM 使用 requests 发送兼容接口请求。
        pass
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return client, local_generator


def normalize_openai_base_url(base_url: str) -> str:
    resolved = base_url.strip().rstrip("/")
    suffix = "/chat/completions"
    if resolved.endswith(suffix):
        return resolved[: -len(suffix)].rstrip("/")
    return resolved


def _compact_response_text(text: str, limit: int = 500) -> str:
    if not text:
        return ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if title_match:
        text = title_match.group(1)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _retry_delay_seconds(response, attempt: int, base_delay: float) -> float:
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return base_delay * attempt


def _raise_api_status_error(response) -> None:
    import requests

    body = _compact_response_text(response.text)
    content_type = response.headers.get("content-type", "")
    ray_id = response.headers.get("cf-ray") or response.headers.get("CF-Ray")
    details = [
        f"{response.status_code} Server Error for url: {response.url}",
        f"content-type={content_type or 'unknown'}",
    ]
    if ray_id:
        details.append(f"cf-ray={ray_id}")
    if body:
        details.append(f"body={body}")
    raise requests.HTTPError("; ".join(details), response=response)


def generate_text(
    prompt: str,
    backend: str,
    api_client: Optional[OpenAI],
    local_generator: Optional[LocalQwenGenerator],
    model: str,
    max_tokens: int,
    response_format: Optional[Dict] = None,
    on_delta: Optional[Callable[[str, str, str], None]] = None,
    stream_id: str = "",
    stream_thinking: bool = False,
    thinking_budget: int = 0,
) -> str:
    if backend in {"openai", "api", "qwen_api"}:
        assert api_client is not None
        stream_requested = on_delta is not None and bool(stream_id)
        request_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0 if max_tokens <= 600 else 0.1,
            "max_tokens": max_tokens,
            "stream": stream_requested,
        }
        if backend == "qwen_api":
            # DashScope 兼容模式：流式时由 STREAM_THINKING 控制，非流式保持关闭
            extra_body = {"enable_thinking": stream_thinking if stream_requested else False}
            if stream_requested and thinking_budget > 0:
                extra_body["thinking_budget"] = thinking_budget
            request_kwargs["extra_body"] = extra_body
        if response_format is not None and backend in {"openai", "qwen_api"}:
            request_kwargs["response_format"] = response_format
        logger.info("LLM request backend=%s model=%s base_url=%s stream_id=%s",
                    backend, model, api_client.base_url, stream_id)
        if stream_requested:
            content_parts: List[str] = []
            stream = api_client.chat.completions.create(**request_kwargs)
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    on_delta(stream_id, "think", reasoning)
                piece = getattr(delta, "content", None)
                if piece:
                    on_delta(stream_id, "answer", piece)
                    content_parts.append(piece)
            return "".join(content_parts)
        response = api_client.chat.completions.create(**request_kwargs)
        return strip_think_section(response.choices[0].message.content)
    assert local_generator is not None
    return local_generator.generate(prompt, max_new_tokens=max_tokens)


THINK_CLOSE_TAG = "</think>"


def strip_think_section(content: str) -> str:
    """部分自部署模型把思考与答案都输出在 content，思考结束带 </think> 标识；非流式响应取标识之后的答案部分。"""
    idx = content.find(THINK_CLOSE_TAG)
    return content[idx + len(THINK_CLOSE_TAG):].strip() if idx >= 0 else content


class _ThinkAnswerSplitter:
    """content 混合思考与答案时按 </think> 标识分流：标识之前的内容归 think，
    之后归 answer；标识可能被切在相邻 delta 之间。部分端点忽略 enable_thinking
    参数，content 形态在「思考+标识+答案」与「直接输出答案」之间漂移，无法在
    流开头预判，因此标识出现前统一下发，流结束时仍未见到标识则把全部缓冲
    归 answer，保证纯答案场景返回值非空。"""

    def __init__(self):
        self.state = "thinking"
        self.pending = ""

    def feed(self, text: str, emit: Callable[[str, str], None]) -> None:
        self.pending += text
        while True:
            idx = self.pending.find(THINK_CLOSE_TAG)
            if idx < 0:
                break
            head = self.pending[:idx]
            self.pending = self.pending[idx + len(THINK_CLOSE_TAG):]
            if head:
                emit("think", head)
            self.state = "answer"
        if self.state == "answer" and self.pending:
            emit("answer", self.pending)
            self.pending = ""

    def flush(self, emit: Callable[[str, str], None]) -> None:
        if not self.pending:
            return
        # 流结束仍未出现标识：本次响应没有思考分离结构，缓冲内容整体作为 answer
        emit("answer", self.pending)
        self.pending = ""


def _consume_requests_stream(response, on_delta: Callable[[str, str, str], None], stream_id: str, stream_thinking: bool) -> str:
    # 消费 OpenAI 兼容 SSE 流。远程端点的 content-type 可能不声明 charset，
    # requests 流式解码会退回 ISO-8859-1 导致中文乱码，因此必须拿原始字节
    # 自行按 UTF-8 增量解码；reasoning_content 归 think；content 在已出现
    # reasoning_content 时归 answer，请求开启思考时按 </think> 标识分离，
    # 关闭思考时直接归 answer，返回答案部分的累积文本供后续解析
    content_parts: List[str] = []
    splitter = _ThinkAnswerSplitter()
    state = {"saw_reasoning": False, "finished": False}

    def emit(kind: str, text: str) -> None:
        if kind == "answer":
            content_parts.append(text)
        on_delta(stream_id, kind, text)

    def handle_line(line: str) -> None:
        line = line.strip()
        if not line.startswith("data:"):
            return  # SSE 注释行（如 ": ping"）直接跳过
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            state["finished"] = True
            return
        if not payload:
            return
        delta = (json.loads(payload).get("choices") or [{}])[0].get("delta") or {}
        reasoning = delta.get("reasoning_content")
        if reasoning:
            state["saw_reasoning"] = True
            emit("think", reasoning)
        piece = delta.get("content")
        if piece:
            if stream_thinking and not state["saw_reasoning"]:
                splitter.feed(piece, emit)
            else:
                emit("answer", piece)

    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    for chunk in response.iter_content(chunk_size=None):
        if state["finished"]:
            break
        if not chunk:
            continue
        buffer += decoder.decode(chunk, False)
        lines = buffer.split("\n")
        buffer = lines.pop()  # 末段可能是被截断的行，留待与下一块拼接
        for line in lines:
            handle_line(line)
            if state["finished"]:
                break
    if not state["finished"]:
        buffer += decoder.decode(b"", True)
        handle_line(buffer)
    splitter.flush(emit)
    return "".join(content_parts)


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
    on_delta: Optional[Callable[[str, str, str], None]] = None,
    stream_id: str = "",
    stream_thinking: bool = False,
    thinking_budget: int = 0,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> str:
    import requests

    stream_requested = on_delta is not None and bool(stream_id)

    if base_url:
        resolved_base_url = normalize_openai_base_url(base_url)
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

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0 if max_tokens <= 600 else 0.1,
        "max_tokens": max_tokens,
        "stream": stream_requested,
    }
    if stream_requested:
        # 流式思考开关：qwen_api（DashScope 兼容端点）用顶层 enable_thinking；
        # 自部署 vllm 端点开启思考会触发不可见的隐藏思考（思考不返回、计入
        # max_tokens 配额且不受 thinking_budget 约束，抽取类请求会被思考吃光
        # 配额），统一按关闭思考请求；模型把思考混入 content 并带 </think>
        # 标识时再由 _consume_requests_stream 分流
        if backend == "qwen_api":
            data["enable_thinking"] = stream_thinking
        if backend == "vllm":
            data["chat_template_kwargs"] = {"enable_thinking": False}
        # 思考强度：0 表示不限制，仅在流式且开启思考时下发
        if stream_thinking and thinking_budget > 0:
            data["thinking_budget"] = thinking_budget
    else:
        if backend == "qwen_api":
            data["enable_thinking"] = False
        if backend == "vllm":
            data["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if response_format is not None and backend in {"openai", "qwen_api"}:
        data["response_format"] = response_format

    url = f"{resolved_base_url}/chat/completions"
    logger.info("LLM request backend=%s model=%s base_url=%s stream_id=%s",
                backend, model, resolved_base_url, stream_id)
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=timeout,
                stream=stream_requested,
            )
        except requests.RequestException:
            if attempt >= max_retries:
                raise
            time.sleep(retry_backoff_seconds * attempt)
            continue

        if not response.ok:
            if response.status_code not in RETRYABLE_HTTP_STATUS_CODES or attempt >= max_retries:
                _raise_api_status_error(response)
            time.sleep(_retry_delay_seconds(response, attempt, retry_backoff_seconds))
            continue

        # 重试时使用带 attempt 标记的 stream_id，避免与中断前已上报的 delta 混淆
        effective_stream_id = stream_id if attempt == 1 else f"{stream_id}#retry{attempt}"
        if stream_requested:
            return _consume_requests_stream(response, on_delta, effective_stream_id, stream_thinking)
        result = response.json()
        return strip_think_section(result["choices"][0]["message"]["content"])
    raise RuntimeError(f"API request failed for url: {url}")
