import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill4re.backends import build_backend, generate_text_with_requests
from skill4re.config import (
    DEFAULT_CHUNK_BUDGET,
    DEFAULT_CHUNK_TRIGGER,
    DEFAULT_INPUT_PATH,
    DEFAULT_LOCAL_MODEL_PATH,
    DEFAULT_MAX_WORKERS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_ROUTE_CACHE_PATH,
    DEFAULT_SKILLS_DIR,
)
from skill4re.dataset import load_samples
from skill4re.loader import load_skills
from skill4re.routing import load_route_cache, save_route_cache
from skill4re.service import SkillRouterExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run modular skill-routed relation extraction.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--chunk-trigger", type=int, default=DEFAULT_CHUNK_TRIGGER)
    parser.add_argument("--chunk-budget", type=int, default=DEFAULT_CHUNK_BUDGET)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--backend", choices=["openai", "api", "qwen_api", "local_qwen3", "vllm"], default="api")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--local-model-path", default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--route-cache", default=str(DEFAULT_ROUTE_CACHE_PATH))
    parser.add_argument("--skills-dir", default=str(DEFAULT_SKILLS_DIR))
    parser.add_argument("--fast-mode", action="store_true", help="使用规则合并替代LLM合并，加快速度")
    parser.add_argument("--skip-coref", action="store_true", help="跳过共指消解，加快速度")
    return parser


def main() -> None:
    run_started_at = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args()
    if args.backend == "vllm":
        api_key = args.api_key or os.getenv("VLLM_API_KEY", "EMPTY")
    else:
        api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if args.backend in {"openai", "api", "qwen_api"} and not api_key:
        parser.error("--api-key is required when --backend openai/api/qwen_api")

    skills = load_skills(Path(args.skills_dir))
    route_cache_path = Path(args.route_cache)
    route_cache = load_route_cache(route_cache_path)

    try:
        client, local_generator = build_backend(
            backend=args.backend,
            api_key=api_key,
            model_path=args.local_model_path,
        )
    except ImportError:
        client = None
        local_generator = None
        print("Warning: openai module not available, using requests fallback")

    extractor = SkillRouterExtractor(
        skills=skills,
        backend=args.backend,
        model=args.model,
        client=client,
        local_generator=local_generator,
        route_cache=route_cache,
        api_key=api_key,
        base_url=args.base_url if args.backend == "vllm" else None,
        fast_mode=args.fast_mode,
        skip_coref=args.skip_coref,
    )

    samples = load_samples(Path(args.input), args.limit)
    results = []
    for sample in samples:
        sample_started_at = time.perf_counter()
        extraction = extractor.extract_document(
            doc_text=sample["doc_text"],
            chunk_trigger=args.chunk_trigger,
            chunk_budget=args.chunk_budget,
            max_workers=args.max_workers,
        )
        sample_wall_seconds = round(time.perf_counter() - sample_started_at, 4)
        results.append(
            {
                "sample_index": sample["sample_index"],
                "sample_id": sample["sample_id"],
                "input_meta": sample["input_meta"],
                "runtime": {
                    "sample_wall_seconds": sample_wall_seconds,
                },
                **extraction,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_wall_seconds = round(time.perf_counter() - run_started_at, 4)
    payload = {
        "meta": {
            "backend": args.backend,
            "model": args.model,
            "sample_count": len(results),
            "total_wall_seconds": total_wall_seconds,
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_route_cache(route_cache_path, route_cache)

    print(output_path)
    print(f"total_wall_seconds={total_wall_seconds}")
    for item in results:
        rels = item["prediction"].get("relation_list", [])
        print(
            f"sample={item['sample_id']} skills={','.join(item['routing']['selected_skills'])} relation_count={len(rels)} total_seconds={item['timing']['total_seconds']} wall_seconds={item['runtime']['sample_wall_seconds']}"
        )


if __name__ == "__main__":
    main()
