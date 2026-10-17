import argparse
from dataclasses import replace
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from app.config import AppConfig
from app.agent.evaluation import aggregate_metrics, replay_file
from app.pipeline import bootstrap_pipeline
from app.model_settings import ModelSettingsStore
from app.types import TaskStatus


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 工具回调并发写入时使用各自的临时文件。
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2))
        temporary = Path(handle.name)
    temporary.replace(path)


def snapshot(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mtime_ns": path.stat().st_mtime_ns}


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="使用真实 OCR、抽取和公共模型配置评估文档校验")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--documents-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source, root = args.documents_dir.resolve(), args.output_dir.resolve()
    env_file = args.env_file.resolve() if args.env_file else None
    os.chdir(Path(__file__).resolve().parents[2] / "backend")
    config = AppConfig.from_env(env_file)
    model_settings_store = ModelSettingsStore(config.storage_root)
    config = replace(config, storage_root=root / "extraction", verification_agent_enabled=False,
                     relation_batch_concurrency=4)
    config.ensure_storage_dirs()
    write_json(root / "configuration.json", config.safe_summary())
    annotations = json.loads((source / "annotations.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    pipeline = bootstrap_pipeline(config, model_settings_store)
    reports, integrity = [], []
    for document in manifest["documents"]:
        identity = document["task_id"]
        pdf = source / (identity + ".pdf")
        baseline = config.storage_root / "results" / identity / "result.json"
        if not baseline.exists():
            payload = {"file_path": str(pdf), "filename": pdf.name, "file_type": 0, "fast_mode": True}
            pipeline.task_store.create_task(TaskStatus(identity, pdf.name, "queued", 0, "queued"), payload)
            pipeline.process_task(identity, payload)
        before = snapshot(baseline)
        metrics_path = root / "replay" / identity / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        else:
            def progress(value):
                write_json(root / "progress" / (identity + "-agent.json"), value)
            metrics = replay_file(baseline, root / "replay", config,
                                  annotations=annotations[identity], progress_callback=progress,
                                  model_settings_store=model_settings_store)
        after = snapshot(baseline)
        assert before == after, "Replay modified the baseline input"
        integrity.append({"document": identity, "before": before, "after": after, "unchanged": before == after})
        reports.append(metrics)
        write_json(root / "evaluation.json", {"documents": reports, "aggregate": aggregate_metrics(reports)})
        write_json(root / "input_integrity.json", integrity)
        print(json.dumps({"document": identity, "status": metrics["status"],
                          "changes": metrics["changes"], "seconds": metrics["elapsed_seconds"]}), flush=True)
    return 0 if all(r["status"] in ("completed", "partial", "skipped") for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
