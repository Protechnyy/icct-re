import argparse
from dataclasses import replace
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit

from flask import send_from_directory
from pypdf import PdfReader, PdfWriter
from redis import Redis
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from shared_llm_check_environment import configure_environment
from app.api import create_app
from app.config import AppConfig
from app.model_settings import ModelSettingsStore, resolve_model_settings
from app.worker import run_worker


def config_for(args):
    configure_environment(args.source_env, args.output_dir)
    config = AppConfig.from_env(args.output_dir / "absent.env")
    if args.redis_db is not None:
        url = urlsplit(config.redis_url)
        config = replace(config, redis_url=urlunsplit(url._replace(path=f"/{args.redis_db}")))
    return config


def wait_until(check, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.2)
    raise TimeoutError("真实服务检查超过时限")


def main():
    parser = argparse.ArgumentParser(description="验证公共模型配置及文档快照的真实请求")
    parser.add_argument("--mode", choices=("check", "api", "worker"), default="check")
    parser.add_argument("--source-env", type=Path, default=ROOT / "backend" / ".env")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "shared_llm_live")
    parser.add_argument("--redis-db", type=int)
    parser.add_argument("--port", type=int, default=15001)
    parser.add_argument("--input", type=Path, default=ROOT / "data/uploads/5ef4fbe30e804dd28edc2aa7a281056e/test.pdf")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.source_env = args.source_env.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = config_for(args)
    config.ensure_storage_dirs()
    if args.mode == "worker":
        run_worker(config)
        return
    if args.mode == "api":
        app = create_app(config)
        app.add_url_rule("/", "check_index", lambda: send_from_directory(ROOT / "frontend/dist", "index.html"))
        app.add_url_rule("/assets/<path:filename>", "check_assets",
                         lambda filename: send_from_directory(ROOT / "frontend/dist/assets", filename))
        app.add_url_rule("/<path:filename>", "check_static",
                         lambda filename: send_from_directory(ROOT / "frontend/dist", filename))
        app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)
        return

    logging.basicConfig(level=logging.INFO)
    settings = resolve_model_settings(config)
    remote = requests.Session()
    models_response = remote.get(settings.base_url + "/models",
                                 headers={"Authorization": "Bearer " + settings.api_key}, timeout=20)
    models_response.raise_for_status()
    available = {item["id"] for item in models_response.json()["data"]}
    alternate = "qwen3.8-flash"
    assert settings.model in available and alternate in available and alternate != settings.model
    redis_url = urlsplit(config.redis_url)
    selected = next(number for number in range(15, 0, -1)
                    if Redis.from_url(urlunsplit(redis_url._replace(path=f"/{number}"))).dbsize() == 0)
    config = replace(config, redis_url=urlunsplit(redis_url._replace(path=f"/{selected}")))
    reservation = Redis.from_url(config.redis_url)
    reservation.set("shared-llm-check-reservation", str(args.output_dir))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    report = {"base_url": base, "redis_db": selected, "default_model": settings.model,
              "alternate_model": alternate, "tasks": [], "switches": []}
    report_path = args.output_dir / "live-check.json"

    def save_report():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    save_report()
    processes, handles = [], []

    def start(mode, label):
        handle = (args.output_dir / (label + ".log")).open("w", encoding="utf-8")
        handles.append(handle)
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--mode", mode,
             "--source-env", str(args.source_env), "--output-dir", str(args.output_dir),
             "--redis-db", str(selected), "--port", str(port)],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        )
        processes.append(process)
        return process

    session = requests.Session()
    session.trust_env = False

    def status(identity):
        response = session.get(base + f"/api/status/{identity}", timeout=10)
        response.raise_for_status()
        value = response.json()
        if value["status"] == "failed":
            raise RuntimeError(value.get("error"))
        return value

    def upload(path):
        with path.open("rb") as handle:
            response = session.post(base + "/api/upload", files={"files": (path.name, handle, "application/pdf")},
                                    data={"stream_thinking": "false",
                                          "split_mode": "paragraph", "paragraph_batch_size": "8"}, timeout=20)
        response.raise_for_status()
        identity = response.json()["tasks"][0]["task_id"]
        report["tasks"].append({"task_id": identity})
        save_report()
        return identity

    def update(model, moment):
        response = session.put(base + "/api/settings/model", json={"base_url": settings.base_url, "model": model}, timeout=20)
        response.raise_for_status()
        assert response.json()["model"] == model
        report["switches"].append({"model": model, "moment": moment})
        save_report()

    def complete(identity, model):
        wait_until(lambda: status(identity)["status"] == "succeeded")
        result = json.loads((config.storage_root / "results" / identity / "result.json").read_text())
        assert result["document_meta"]["llm"]["model"] == model
        assert len(result["relation_batches"]) >= 2
        record = next(item for item in report["tasks"] if item["task_id"] == identity)
        record.update({"model": model, "batches": len(result["relation_batches"]),
                       "relations": len(result["final_relations"])})
        save_report()
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return result

    try:
        start("api", "api")
        wait_until(lambda: (args.output_dir / "api.log").exists() and
                   "Running on" in (args.output_dir / "api.log").read_text(), timeout=30)
        response = session.get(base + "/api/settings/model", timeout=10)
        response.raise_for_status()
        assert response.json()["default_model"] == settings.model
        assert "api_key" not in response.json()
        writer = PdfWriter()
        writer.add_page(PdfReader(args.input).pages[0])
        document = args.output_dir / "source-first-page.pdf"
        writer.write(document)
        source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
        first = upload(document)
        second = upload(document)
        start("worker", "worker-1")
        wait_until(lambda: status(first)["stage"] == "layout_parsing")
        assert status(second)["status"] == "queued"
        update(alternate, "during_ocr_and_second_document_queued")
        start("worker", "worker-2")
        complete(first, settings.model)
        complete(second, alternate)
        third = upload(document)
        wait_until(lambda: status(third)["stage"] == "relation_extraction")
        update(settings.model, "during_extraction")
        fourth = upload(document)
        complete(third, alternate)
        complete(fourth, settings.model)
        assert hashlib.sha256(args.input.read_bytes()).hexdigest() == source_hash
        assert ModelSettingsStore(config.storage_root).get(settings.base_url, settings.api_key, settings.model).api_key == settings.api_key
        response = session.put(base + "/api/settings/model", json={"base_url": settings.base_url, "model": "", "api_key": ""}, timeout=20)
        response.raise_for_status()
        assert response.json()["model"] == settings.model
        report["source_unchanged"] = True
        report["success"] = True
        save_report()
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        for process in reversed(processes):
            process.terminate()
            process.wait(timeout=30)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
