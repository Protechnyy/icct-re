import argparse
import os

import httpx
from openai import OpenAI


def build_client(api_key: str, base_url: str, timeout: float = 120.0) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(trust_env=False, timeout=timeout),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Qwen API smoke test")
    parser.add_argument("--model", default="qwen3-32b", help="Model name")
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--prompt",
        default="你好，请用一句话介绍你自己。",
        help="Prompt to send",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY or DASHSCOPE_API_KEY first.")

    client = build_client(api_key=api_key, base_url=args.base_url)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        stream=False,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body={"enable_thinking": False},
    )

    content = response.choices[0].message.content
    print("=== Qwen API Response ===")
    print(content)


if __name__ == "__main__":
    main()
