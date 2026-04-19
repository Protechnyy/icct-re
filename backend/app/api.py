from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from .config import AppConfig
from .paddle_ocr import PaddleOcrClient
from .task_store import RedisTaskStore
from .types import TaskStatus
from .utils import detect_file_type, generate_task_id, safe_filename, utcnow_iso
from .vllm_client import VllmClient

LOGGER = logging.getLogger(__name__)


def create_app() -> Flask:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    config = AppConfig.from_env()
    config.ensure_storage_dirs()

    app = Flask(__name__)
    app.config["APP_CONFIG"] = config
    CORS(app)

    task_store = RedisTaskStore(config.redis_url)
    ocr_client = PaddleOcrClient(config)
    vllm_client = VllmClient(config)

    LOGGER.info("API config loaded: %s", config.safe_summary())

    @app.post("/api/upload")
    def upload() -> tuple[object, int]:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        tasks = []
        for file_storage in files:
            filename = safe_filename(file_storage.filename or "upload")
            task_id = generate_task_id()
            target = _target_path(config.storage_root, task_id, filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            file_storage.save(target)
            file_type = detect_file_type(target)
            status = TaskStatus(
                task_id=task_id,
                filename=filename,
                status="queued",
                progress=0,
                stage="queued",
                created_at=utcnow_iso(),
                updated_at=utcnow_iso(),
            )
            payload = {"file_path": str(target), "filename": filename, "file_type": file_type}
            task_store.create_task(status, payload)
            task_store.enqueue_task(task_id)
            tasks.append(status.to_dict())
            LOGGER.info("Queued task %s for %s -> %s", task_id, filename, target)
        return jsonify({"tasks": tasks}), 202

    @app.get("/api/status/<task_id>")
    def status(task_id: str) -> tuple[object, int]:
        task = task_store.get_task(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        task.pop("payload", None)
        return jsonify(task), 200

    @app.get("/api/result/<task_id>")
    def result(task_id: str) -> tuple[object, int]:
        result_payload = task_store.get_result(task_id)
        if not result_payload:
            task = task_store.get_task(task_id)
            if not task:
                return jsonify({"error": "Task not found"}), 404
            return jsonify({"error": "Result not ready"}), 409
        return jsonify(result_payload), 200

    @app.get("/api/health")
    def health() -> tuple[object, int]:
        dependencies = {
            "redis": task_store.healthcheck(),
            "paddle_ocr": ocr_client.healthcheck(),
            "vllm": vllm_client.healthcheck(),
        }
        health_info = {
            "status": "ok" if all(dependencies.values()) else "degraded",
            **dependencies,
        }
        status_code = 200 if all(dependencies.values()) else 503
        return jsonify(health_info), status_code

    return app


def _target_path(storage_root: Path, task_id: str, filename: str) -> Path:
    return storage_root / "uploads" / task_id / filename
