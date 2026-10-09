#!/usr/bin/env python3
"""Replay only document verification; no OCR or task queue is started."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.config import AppConfig, normalize_openai_base_url
from app.agent.evaluation import aggregate_metrics, replay_file, score_judgments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--annotations", type=Path, help="JSON relation list (one input), or map of task_id to relation lists")
    parser.add_argument("--judgments-csv", nargs="+", type=Path, help="Read completed changes.csv files and score judgments")
    args = parser.parse_args()
    if args.judgments_csv:
        print(json.dumps(score_judgments(args.judgments_csv), ensure_ascii=False, indent=2))
        return 0
    if not args.inputs or args.output_dir is None:
        parser.error("Provide result.json input(s) and --output-dir, or --judgments-csv.")
    config = AppConfig.from_env()
    if args.base_url:
        config = replace(config, agent_base_url=normalize_openai_base_url(args.base_url))
    if args.model:
        config = replace(config, agent_model=args.model)
    if not config.agent_api_key:
        parser.error("Configure AGENT_API_KEY for the independent verification model.")
    annotations = json.loads(args.annotations.read_text()) if args.annotations else None
    if isinstance(annotations, list) and len(args.inputs) != 1:
        parser.error("Multiple inputs require annotations indexed by task_id.")
    reports = []
    for source in args.inputs:
        gold = annotations
        if isinstance(annotations, dict):
            payload = json.loads(source.read_text())
            identity = str(payload.get("document_meta", {}).get("task_id") or source.parent.name)
            gold = annotations.get(identity)
        last_sequence = [None]
        def report_progress(progress):
            events = progress.get("recent_events", [])
            if not events or events[-1]["seq"] == last_sequence[0]:
                return
            event = events[-1]
            last_sequence[0] = event["seq"]
            if event["kind"] in ("plan", "task_end"):
                print(f"{source.parent.name}: {progress['completed_tasks']}/{progress['total_tasks']} {event['kind']}", flush=True)
        report = replay_file(source, args.output_dir, config, annotations=gold, progress_callback=report_progress)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    summary = {"documents": reports, "aggregate": aggregate_metrics(reports)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": summary["aggregate"]}, ensure_ascii=False, indent=2))
    return 0 if all(r["status"] in ("completed", "partial", "skipped") for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
