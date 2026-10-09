#!/usr/bin/env python3
"""One-off Bailian/LangGraph tool-loop probe; never prints credentials or reasoning.

Run with backend/.venv/bin/python scripts/probe-agent-connection.py
  --base-url <Bailian compatible-mode/v1> --model <model-id>
Compare --thinking default and --thinking disabled before choosing adapter settings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.config import _load_dotenv, normalize_openai_base_url


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("AGENT_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL"))
    parser.add_argument("--thinking", choices=("default", "disabled"), default="default")
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    key = os.getenv("AGENT_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not args.base_url or not args.model or not key:
        parser.error("Provide --base-url, --model and AGENT_API_KEY (or DASHSCOPE_API_KEY).")
    # This diagnostic must not upload traces, prompts or model reasoning elsewhere.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    secret_value = "probe-" + secrets.token_hex(4)
    calls = []

    @tool
    def read_probe_value(label: str) -> str:
        """Read the value for the diagnostic label 'connection-test'."""
        calls.append(label)
        return secret_value if label == "connection-test" else "unknown label"

    started = time.monotonic()
    report = {"model": args.model, "thinking_parameter": args.thinking}
    try:
        model = ChatOpenAI(
            model=args.model, api_key=key,
            base_url=normalize_openai_base_url(args.base_url),
            timeout=args.timeout, max_retries=0, max_tokens=1024,
            extra_body={"enable_thinking": False} if args.thinking == "disabled" else {},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            graph = create_react_agent(model, [read_probe_value], prompt=(
                "Call read_probe_value with label connection-test exactly once. "
                "Then return the value from the tool as your final answer. Do not guess it."
            ))
        result = graph.invoke({"messages": [("user", "Read the diagnostic value.")]},
                              {"recursion_limit": 6})
        messages = result["messages"]
        answers = [message for message in messages if isinstance(message, AIMessage)]
        final = answers[-1]
        usage = [message.usage_metadata for message in answers]
        report.update({
            "tool_calls": [call for message in answers for call in message.tool_calls],
            "tool_results": [message.content for message in messages if isinstance(message, ToolMessage)],
            "final_answer": final.content,
            "tool_executed": bool(calls),
            "final_uses_tool_result": bool(calls) and secret_value in str(final.content),
            "reasoning_returned": any(bool(message.additional_kwargs.get("reasoning_content"))
                                      for message in answers),
            "usage": usage,
            "usage_returned": bool(usage) and all(item is not None for item in usage),
            "response_models": [message.response_metadata.get("model_name") for message in answers],
        })
        success = report["final_uses_tool_result"]
    except Exception as exc:
        # Only report the class/status: provider exception text may include request data.
        report.update({"error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)})
        success = False
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["success"] = success
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
