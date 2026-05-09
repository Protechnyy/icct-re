import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import Pipeline
import models


DEFAULT_INSTRUCTION = (
    "请从给定作战任务文档中抽取关键关系三元组。"
    "优先抽取文档级高价值关系，例如单位与职责、单位与部署位置、敌我控制关系、"
    "作战目标、时间约束、失败条件、节点作用、链路压制等。"
    "请为每条关系提供原文证据。"
)


def load_samples(path: Path, limit: int):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if len(samples) >= limit:
                break
            obj = json.loads(line)
            doc = obj.get("output", "")
            if not doc:
                continue
            samples.append(
                {
                    "sample_index": idx,
                    "sample_id": obj.get("sample_id"),
                    "input_meta": obj.get("input", {}),
                    "doc_text": doc,
                }
            )
    return samples


def main():
    parser = argparse.ArgumentParser(description="Run OneKE Base-mode trial on fight_data.jsonl")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "fight_data.jsonl"),
        help="Path to the jsonl file.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "examples" / "results" / "fight_data_base_trial.json"),
        help="Path to save the extraction results.",
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to run.")
    parser.add_argument("--model-name", default="deepseek-chat", help="API model name.")
    parser.add_argument("--api-key", required=True, help="DeepSeek API key.")
    args = parser.parse_args()

    model = models.DeepSeek(args.model_name, args.api_key, "https://api.deepseek.com")
    pipeline = Pipeline(model)

    samples = load_samples(Path(args.input), args.limit)
    results = []

    for sample in samples:
        result, trajectory, _, _ = pipeline.get_extract_result(
            task="Base",
            instruction=DEFAULT_INSTRUCTION,
            text=sample["doc_text"],
            output_schema="",
            constraint="",
            use_file=False,
            file_path="",
            truth="",
            mode="quick",
            update_case=False,
            show_trajectory=False,
            config_name="",
        )
        results.append(
            {
                "sample_index": sample["sample_index"],
                "sample_id": sample["sample_id"],
                "input_meta": sample["input_meta"],
                "instruction": DEFAULT_INSTRUCTION,
                "trajectory": trajectory,
                "prediction": result,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(output_path)
    print(f"samples={len(results)}")
    for item in results:
        pred = item["prediction"]
        relation_count = len(pred.get("relation_list", [])) if isinstance(pred, dict) else 0
        print(f"sample={item['sample_id']} relation_count={relation_count}")


if __name__ == "__main__":
    main()
