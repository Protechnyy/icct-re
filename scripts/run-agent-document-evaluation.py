#!/usr/bin/env python3
"""Run live OCR, baseline extraction and independent replay of generated PDFs.

This makes real network calls. Configure extraction and AGENT_API_KEY first.
Existing baseline/replay results are reused; remove the output directory to rerun.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.config import AppConfig
from app.agent.evaluation import aggregate_metrics, replay_file
from app.paddle_ocr import PaddleOcrClient
from app.pipeline import DocumentPipeline
from app.skill4re_client import Skill4ReClient


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Tool callbacks can run concurrently; each write needs its own temporary file.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2))
        temporary = Path(handle.name)
    temporary.replace(path)


class FileTaskStore:
    """Persist progress without starting a Redis worker or web server."""
    def __init__(self, root):
        self.root = root
        self.status = {}

    def update_task(self, task_id, **updates):
        self.status.setdefault(task_id, {}).update(updates)
        write_json(self.root / (task_id + ".json"), self.status[task_id])
        print(task_id, updates.get("stage", ""), flush=True)

    def set_result(self, task_id, result):
        pass  # DocumentPipeline itself saves the full result.


def snapshot(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mtime_ns": path.stat().st_mtime_ns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source, root = args.documents_dir.resolve(), args.output_dir.resolve()
    # .env paths in this project are relative to the backend startup directory.
    os.chdir(Path(__file__).resolve().parents[1] / "backend")
    config = AppConfig.from_env()
    if not config.agent_api_key:
        parser.error("Configure AGENT_API_KEY for the live verification run.")
    config = replace(config, storage_root=root / "extraction", agent_enabled=False,
                     relation_batch_concurrency=4)
    config.ensure_storage_dirs()
    write_json(root / "configuration.json", config.safe_summary())
    annotations = json.loads((source / "annotations.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    store = FileTaskStore(root / "progress")
    pipeline = DocumentPipeline(config, store, PaddleOcrClient(config), Skill4ReClient(config))
    reports, integrity = [], []
    for document in manifest["documents"]:
        identity = document["task_id"]
        pdf = source / (identity + ".pdf")
        baseline = config.storage_root / "results" / identity / "result.json"
        if not baseline.exists():
            pipeline.process_task(identity, {"file_path": str(pdf), "filename": pdf.name, "file_type": 0})
        before = snapshot(baseline)
        metrics_path = root / "replay" / identity / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        else:
            def progress(value):
                write_json(root / "progress" / (identity + "-agent.json"), value)
            metrics = replay_file(baseline, root / "replay", config,
                                  annotations=annotations[identity], progress_callback=progress)
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
