import json
import os
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI


ROOT = Path("/home/users/lhy/OneKE")
INPUT_PATH = ROOT / "data" / "fight_data.jsonl"
OUTPUT_PATH = ROOT / "examples" / "results" / "fight_data_base_qwen3_32b_reflect.json"
API_KEY = os.environ.get("OPENAI_API_KEY", "")

sys.path.insert(0, str(ROOT / "src"))

from pipeline import Pipeline
from models.llm_def import ChatGPT


INSTRUCTION = (
    "请以开放域方式抽取给定军事文档中所有有意义的军事关系，并严格按固定 schema 输出。"
    "输出格式必须为 {\"relation_list\": [...]}。每条关系都必须包含 head、relation、tail、evidence、skill 五个字段。"
    "其中 skill 只能从 force-organization、operation-constraint、engagement-effects 中选择。"
    "重点覆盖兵力编成与部署、指挥控制、阶段任务与时间、失败条件与约束、侦察链路、火力作用、电子干扰、桥梁/道路/补给线等关键节点作用关系。"
    "不要输出摘要式描述，不要漏掉有直接证据支持的具体关系，只返回结构化 JSON。"
)


class Qwen3Chat(ChatGPT):
    def __init__(self, model_name_or_path: str, api_key: str, base_url: str):
        self.name = "ChatGPT"
        self.model = model_name_or_path
        self.base_url = base_url
        self.temperature = 0.2
        self.top_p = 0.9
        self.max_tokens = 4096
        self.api_key = api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.Client(trust_env=False, timeout=120.0),
        )
        self.request_count = 0

    def get_chat_response(self, input_text: str):
        self.request_count += 1
        request_id = self.request_count
        prompt_chars = len(input_text)
        preview = input_text[:160].replace("\n", " ")
        started_at = time.perf_counter()
        print(
            f"[Qwen3Chat] request #{request_id} start | prompt_chars={prompt_chars} | preview={preview}",
            flush=True,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": input_text}],
            stream=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=None,
            extra_body={"enable_thinking": False},
        )
        elapsed = time.perf_counter() - started_at
        content = response.choices[0].message.content
        output_chars = len(content) if content else 0
        print(
            f"[Qwen3Chat] request #{request_id} done  | elapsed={elapsed:.2f}s | output_chars={output_chars}",
            flush=True,
        )
        return content


def main():
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    total_started_at = time.perf_counter()
    item = json.loads(INPUT_PATH.read_text(encoding="utf-8").splitlines()[0])
    text = item["output"]
    model = Qwen3Chat(
        model_name_or_path="qwen3-32b",
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    pipeline = Pipeline(model)
    result, trajectory, schema, _ = pipeline.get_extract_result(
        task="Base",
        three_agents={},
        instruction=INSTRUCTION,
        text=text,
        output_schema="MilitaryRelationList",
        constraint="",
        mode={
            "schema_agent": "get_retrieved_schema",
            "extraction_agent": "extract_information_with_case",
            "reflection_agent": "reflect_with_case",
        },
        update_case=False,
        show_trajectory=False,
        verbose=True,
    )
    total_elapsed = time.perf_counter() - total_started_at
    output = {
        "model": "qwen3-32b",
        "task": "Base",
        "mode": "custom-fixed-schema-with-reflection",
        "instruction": INSTRUCTION,
        "elapsed_seconds": round(total_elapsed, 4),
        "schema": schema,
        "trajectory": trajectory,
        "result": result,
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
