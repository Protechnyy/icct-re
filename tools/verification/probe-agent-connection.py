from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import warnings

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from app.agent.document import DocumentIndex
from app.config import AppConfig
from app.model_settings import resolve_model_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="使用公共模型连接和真实文档查询工具检查连接")
    parser.add_argument("result", type=Path, help="包含原文及小节的 result.json")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--thinking", choices=("default", "disabled"), default="disabled")
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args()
    settings = resolve_model_settings(AppConfig.from_env(args.env_file))
    settings = replace(settings, base_url=args.base_url or settings.base_url,
                       model=args.model or settings.model,
                       request_timeout_seconds=args.timeout or settings.request_timeout_seconds)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    index = DocumentIndex(payload["relation_sections"], payload["document_text"])
    section_id = next(iter(index.sections))
    calls = []

    def read_section(section_id: str) -> dict:
        result = index.read_section(section_id)
        calls.append(result)
        return result

    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    model = ChatOpenAI(model=settings.model, api_key=lambda: settings.api_key,
                       base_url=settings.base_url, timeout=settings.request_timeout_seconds,
                       max_retries=0, max_tokens=1024,
                       extra_body=settings.thinking_options(False) if args.thinking == "disabled" else {})
    tool = StructuredTool.from_function(read_section, name="read_section", description="读取当前文档的小节原文")
    started = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        graph = create_react_agent(model, [tool], prompt="调用 read_section 读取指定小节，然后根据原文给出一句简短说明。")
    result = graph.invoke({"messages": [("user", f"请读取小节 {section_id}。") ]}, {"recursion_limit": 6})
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert calls and tool_messages and calls[0] == index.read_section(section_id)
    report = {**settings.public_summary(), "tool_executed": True,
              "tool": "read_section", "section_id": section_id,
              "elapsed_seconds": round(time.monotonic() - started, 3), "success": True}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
