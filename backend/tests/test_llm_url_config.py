from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from app.config import normalize_openai_base_url
from app.vllm_client import VllmClient


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL4RE_ROOT = REPO_ROOT / "skill4re"
if str(SKILL4RE_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL4RE_ROOT))

from skill4re.backends import (  # noqa: E402
    generate_text_with_requests,
    normalize_openai_base_url as normalize_skill4re_base_url,
)


class FakeResponse:
    ok = True
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "[]"}}]}


class RecordingSession:
    def __init__(self, get_response: FakeResponse | None = None) -> None:
        self.get_response = get_response or FakeResponse()
        self.get_urls: list[str] = []
        self.post_urls: list[str] = []

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.get_urls.append(url)
        return self.get_response

    def post(self, url: str, **kwargs) -> FakeResponse:
        self.post_urls.append(url)
        return FakeResponse()


def test_normalize_openai_base_url_accepts_full_chat_completions_url() -> None:
    url = "https://api.asukalangely.top/v1/chat/completions"

    assert normalize_openai_base_url(url) == "https://api.asukalangely.top/v1"
    assert normalize_skill4re_base_url(url) == "https://api.asukalangely.top/v1"


def test_vllm_client_posts_to_single_chat_completions_path() -> None:
    config = SimpleNamespace(
        vllm_base_url="https://api.asukalangely.top/v1/chat/completions",
        vllm_api_key="secret",
        vllm_model="Qwen3-32B-BF16",
        vllm_timeout_seconds=10,
        vllm_enable_thinking=False,
    )
    session = RecordingSession()
    client = VllmClient(config, session=session)

    assert client.healthcheck() is True
    assert client.extract_relations("Alice joined ACME.", "test") == []
    assert session.get_urls == ["https://api.asukalangely.top/v1/models"]
    assert session.post_urls == ["https://api.asukalangely.top/v1/chat/completions"]


def test_vllm_healthcheck_falls_back_to_chat_when_models_is_unavailable() -> None:
    config = SimpleNamespace(
        vllm_base_url="https://api.asukalangely.top/v1",
        vllm_api_key="secret",
        vllm_model="Qwen3-32B-BF16",
        vllm_timeout_seconds=10,
        vllm_enable_thinking=False,
    )
    models_response = FakeResponse()
    models_response.ok = False
    models_response.status_code = 404
    session = RecordingSession(get_response=models_response)
    client = VllmClient(config, session=session)

    assert client.healthcheck() is True
    assert session.get_urls == ["https://api.asukalangely.top/v1/models"]
    assert session.post_urls == ["https://api.asukalangely.top/v1/chat/completions"]


class ApiResponse:
    def __init__(self, status_code: int, text: str = "", content: str = "ok") -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = text
        self.content = content
        self.url = "https://api.asukalangely.top/v1/chat/completions"
        self.headers = {"content-type": "text/html; charset=UTF-8"}

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self.content}}]}


def test_skill4re_requests_retries_transient_bad_gateway(monkeypatch) -> None:
    responses = [
        ApiResponse(502, "<html><title>api.asukalangely.top | 502: Bad gateway</title></html>"),
        ApiResponse(200, content='{"relation_list": []}'),
    ]
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return responses.pop(0)

    monkeypatch.setenv("VLLM_MAX_RETRIES", "2")
    monkeypatch.setenv("VLLM_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    content = generate_text_with_requests(
        prompt="test",
        api_key="secret",
        model="Qwen3-32B-BF16",
        max_tokens=16,
        backend="vllm",
        base_url="https://api.asukalangely.top/v1/chat/completions",
    )

    assert content == '{"relation_list": []}'
    assert calls == [
        "https://api.asukalangely.top/v1/chat/completions",
        "https://api.asukalangely.top/v1/chat/completions",
    ]


def test_skill4re_requests_raises_compact_error_after_retries(monkeypatch) -> None:
    import requests

    html = "<html><title>api.asukalangely.top | 502: Bad gateway</title><body>" + ("x" * 2000) + "</body></html>"

    def fake_post(url, **kwargs):
        return ApiResponse(502, html)

    monkeypatch.setenv("VLLM_MAX_RETRIES", "1")
    monkeypatch.setattr("requests.post", fake_post)

    try:
        generate_text_with_requests(
            prompt="test",
            api_key="secret",
            model="Qwen3-32B-BF16",
            max_tokens=16,
            backend="vllm",
            base_url="https://api.asukalangely.top/v1",
        )
    except requests.HTTPError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected HTTPError")

    assert "502" in message
    assert "Bad gateway" in message
    assert "<html>" not in message
    assert len(message) < 700
