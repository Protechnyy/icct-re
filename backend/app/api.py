from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from urllib.parse import quote, urlsplit

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from redis.exceptions import ConnectionError as RedisConnectionError

from .config import AppConfig, RELATION_SPLIT_MODES
from .llm_stream import decode_entry, stream_key
from .model_settings import ModelSettingsError, ModelSettingsStore, main_model_settings_defaults
from .paddle_ocr import PaddleOcrClient
from .skill_store import SkillStore, SkillStoreError
from .task_store import RedisTaskStore
from .types import TaskStatus
from .utils import detect_file_type, ensure_supported_file_suffix, generate_task_id, safe_filename, utcnow_iso
from .vllm_client import VllmClient

LOGGER = logging.getLogger(__name__)
STREAM_ENTRY_ID_RE = re.compile(r"^\d+(-\d+)?$")


def _sse_event(event_name: str, data: str, seq: str) -> str:
    return f"event: {event_name}\ndata: {data}\nid: {seq}\n\n"


def create_app() -> Flask:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    config = AppConfig.from_env()
    config.ensure_storage_dirs()

    app = Flask(__name__)
    app.config["APP_CONFIG"] = config
    app.json.sort_keys = False
    CORS(app)

    task_store = RedisTaskStore(config.redis_url)
    ocr_client = PaddleOcrClient(config)
    model_settings_store = ModelSettingsStore(config.storage_root)
    vllm_client = VllmClient(config, settings_store=model_settings_store)
    skill_store = SkillStore(config)

    LOGGER.info("API config loaded: %s", config.safe_summary())

    @app.post("/api/upload")
    def upload() -> tuple[object, int]:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files uploaded"}), 400
        try:
            relation_split_config = _relation_split_config_from_form(request.form, config)
            fast_mode_value = request.form.get("fast_mode")
            if fast_mode_value is not None and fast_mode_value not in {"true", "false"}:
                raise ValueError("fast_mode must be true or false")
            fast_mode = fast_mode_value == "true" if fast_mode_value is not None else not config.agent_enabled
            stream_thinking_value = request.form.get("stream_thinking")
            if stream_thinking_value is not None and stream_thinking_value not in {"true", "false"}:
                raise ValueError("stream_thinking must be true or false")
            stream_thinking = stream_thinking_value == "true" if stream_thinking_value is not None else config.stream_thinking
            thinking_budget_value = request.form.get("thinking_budget")
            if thinking_budget_value is None or thinking_budget_value == "":
                thinking_budget = 0
            else:
                thinking_budget = int(thinking_budget_value)
                if thinking_budget < 0 or thinking_budget > 99999:
                    raise ValueError("thinking_budget must be between 0 and 99999")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        tasks = []
        for file_storage in files:
            filename = safe_filename(file_storage.filename or "upload")
            task_id = generate_task_id()
            target = _target_path(config.storage_root, task_id, filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            file_storage.save(target)
            try:
                target = ensure_supported_file_suffix(target)
                file_type = detect_file_type(target)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            status = TaskStatus(
                task_id=task_id,
                filename=filename,
                status="queued",
                progress=0,
                stage="queued",
                created_at=utcnow_iso(),
                updated_at=utcnow_iso(),
                metadata={"split_mode": relation_split_config["split_mode"], "fast_mode": fast_mode},
            )
            payload = {
                "file_path": str(target),
                "filename": filename,
                "stored_filename": target.name,
                "file_type": file_type,
                "relation_split_config": relation_split_config,
                "fast_mode": fast_mode,
                "stream_thinking": stream_thinking,
                "thinking_budget": thinking_budget,
            }
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

    @app.get("/api/agent/<task_id>/events")
    def agent_events(task_id: str) -> tuple[object, int]:
        after_seq_text = request.args.get("after_seq", "0")
        limit_text = request.args.get("limit", "100")
        if not after_seq_text.isdecimal() or not limit_text.isdecimal():
            return jsonify({"error": "after_seq 和 limit 必须是整数"}), 400
        after_seq, limit = int(after_seq_text), int(limit_text)
        if not 1 <= limit <= 200:
            return jsonify({"error": "limit 必须介于 1 和 200"}), 400
        result = task_store.get_agent_events(task_id, after_seq, limit)
        if result is None:
            return jsonify({"error": "文档任务不存在"}), 404
        return jsonify(result), 200

    @app.get("/api/agent/<task_id>/events/<seq>")
    def agent_event(task_id: str, seq: str) -> tuple[object, int]:
        if not seq.isdecimal() or int(seq) < 1:
            return jsonify({"error": "事件序号必须是正整数"}), 400
        if task_store.get_task(task_id) is None:
            return jsonify({"error": "文档任务不存在"}), 404
        event = task_store.get_agent_event(task_id, int(seq))
        if event is None:
            return jsonify({"error": "执行事件不存在"}), 404
        return jsonify(event), 200

    @app.get("/api/tasks/<task_id>/stream")
    def task_stream(task_id: str) -> Response | tuple[object, int]:
        if task_store.get_task(task_id) is None:
            return jsonify({"error": "Task not found"}), 404
        # 连接建立前探测 Redis，不可用时以 503 返回，避免进入 SSE 后无法表达错误
        try:
            task_store.redis.ping()
        except RedisConnectionError:
            return jsonify({"error": "Redis unavailable"}), 503
        # EventSource 自动重连携带 Last-Event-ID，优先于查询参数
        after_seq = request.headers.get("Last-Event-ID") or request.args.get("after_seq", "0")
        after_seq = str(after_seq).strip()
        if not STREAM_ENTRY_ID_RE.match(after_seq):
            return jsonify({"error": "after_seq must be a stream entry id"}), 400
        cursor = after_seq if "-" in after_seq else "0-0"

        def generate(cursor: str):
            while True:
                response = task_store.redis.xread(
                    {stream_key(task_id): cursor}, block=15000, count=200,
                )
                if not response:
                    # 空闲期心跳注释行，维持连接不被代理与浏览器掐断
                    yield ": ping\n\n"
                    continue
                for _, entries in response:
                    for entry_id, fields in entries:
                        cursor = entry_id
                        event = decode_entry(entry_id, fields)
                        payload = json.dumps(event, ensure_ascii=False)
                        yield _sse_event("delta", payload, entry_id)
                        if event["kind"] == "done":
                            yield _sse_event("end", payload, entry_id)
                            return

        response = Response(generate(cursor), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.get("/api/result/<task_id>/csv")
    def result_csv(task_id: str) -> Response | tuple[object, int]:
        task = task_store.get_task(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        result_payload = task_store.get_result(task_id)
        if not result_payload:
            return jsonify({"error": "Result not ready"}), 409

        source_name = Path(str(task.get("filename") or task_id)).stem
        download_name = f"{source_name}_relations.csv"
        response = Response(_relations_to_csv(result_payload), content_type="text/csv; charset=utf-8")
        response.headers["Content-Disposition"] = (
            f'attachment; filename="relations.csv"; filename*=UTF-8\'\'{quote(download_name)}'
        )
        response.headers["Cache-Control"] = "no-store"
        return response

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

    @app.get("/api/settings/model")
    def get_model_settings() -> tuple[object, int]:
        model_defaults = main_model_settings_defaults(config)
        if model_defaults is None:
            return jsonify({"error": "当前主模型使用本地后端，远程 API 设置不可用"}), 409
        try:
            settings = model_settings_store.get(*model_defaults)
        except ModelSettingsError as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({
            "base_url": settings.base_url,
            "model": config.skill4re_model,
            "backend": config.skill4re_backend,
            "api_key_configured": settings.api_key_configured,
        }), 200

    @app.put("/api/settings/model")
    def update_model_settings() -> tuple[object, int]:
        if not _model_settings_write_allowed():
            return jsonify({"error": "模型设置只能从本机或同源页面更新"}), 403
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "请求内容必须是 JSON 对象"}), 400
        base_url = payload.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            return jsonify({"error": "API 地址不能为空"}), 400
        model_defaults = main_model_settings_defaults(config)
        if model_defaults is None:
            return jsonify({"error": "当前主模型使用本地后端，远程 API 设置不可用"}), 409
        try:
            current = model_settings_store.get(*model_defaults)
            if "api_key" in payload:
                api_key_value = payload["api_key"]
                if not isinstance(api_key_value, str):
                    return jsonify({"error": "API Key 必须是文本"}), 400
                api_key = api_key_value.strip() or current.api_key
            else:
                api_key = current.api_key
            settings = model_settings_store.save(base_url, api_key)
        except ModelSettingsError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({
            "base_url": settings.base_url,
            "model": config.skill4re_model,
            "backend": config.skill4re_backend,
            "api_key_configured": settings.api_key_configured,
        }), 200

    @app.get("/api/skills")
    def list_skills() -> tuple[object, int]:
        try:
            skills = skill_store.list_skills()
        except SkillStoreError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"skills": skills, "skills_dir": str(config.skill4re_skills_dir)}), 200

    @app.get("/api/skills/<skill_name>")
    def get_skill(skill_name: str) -> tuple[object, int]:
        try:
            return jsonify(skill_store.get_skill(skill_name)), 200
        except FileNotFoundError:
            return jsonify({"error": "Skill not found"}), 404
        except SkillStoreError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/skills")
    def create_skill() -> tuple[object, int]:
        payload = request.get_json(silent=True)
        try:
            created = skill_store.create_skill(payload if isinstance(payload, dict) else {})
        except FileExistsError as exc:
            return jsonify({"error": str(exc)}), 409
        except SkillStoreError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(created), 201

    @app.put("/api/skills/<skill_name>")
    def update_skill(skill_name: str) -> tuple[object, int]:
        payload = request.get_json(silent=True)
        try:
            updated = skill_store.update_skill(skill_name, payload if isinstance(payload, dict) else {})
        except FileNotFoundError:
            return jsonify({"error": "Skill not found"}), 404
        except FileExistsError as exc:
            return jsonify({"error": str(exc)}), 409
        except SkillStoreError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(updated), 200

    return app


def _target_path(storage_root: Path, task_id: str, filename: str) -> Path:
    return storage_root / "uploads" / task_id / filename


def _model_settings_write_allowed() -> bool:
    origin = request.headers.get("Origin")
    if not origin:
        return request.remote_addr in {"127.0.0.1", "::1"}
    parsed_origin = urlsplit(origin)
    origin_host = parsed_origin.hostname
    request_host = urlsplit(f"{request.scheme}://{request.host}").hostname
    return bool(
        parsed_origin.scheme in {"http", "https"}
        and origin_host
        and (origin_host in {"localhost", "127.0.0.1", "::1"} or origin_host == request_host)
    )


CSV_COLUMNS = (
    "主体",
    "关系",
    "客体",
    "证据",
    "Skill",
    "来源章节",
    "来源页码",
    "来源块",
    "来源段落",
    "来源批次",
)


def _relations_to_csv(result_payload: dict[str, object]) -> str:
    relation_container = result_payload.get("final_relation_list")
    if isinstance(relation_container, dict):
        relations = relation_container.get("relation_list", [])
    else:
        relations = result_payload.get("final_relations", [])
    if not isinstance(relations, list):
        relations = []

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        writer.writerow(
            {
                "主体": _csv_text(relation.get("head") or relation.get("subject")),
                "关系": _csv_text(relation.get("relation")),
                "客体": _csv_text(relation.get("tail") or relation.get("object")),
                "证据": _csv_text(relation.get("evidence")),
                "Skill": _csv_text(relation.get("skill")),
                "来源章节": _csv_list(relation.get("source_sections")),
                "来源页码": _csv_list(relation.get("source_pages")),
                "来源块": _csv_list(relation.get("source_blocks")),
                "来源段落": _csv_paragraphs(relation.get("source_paragraphs")),
                "来源批次": _csv_text(relation.get("source_batch_index")),
            }
        )
    return "\ufeff" + output.getvalue()


def _csv_list(value: object) -> str:
    values = value if isinstance(value, list) else ([] if value in (None, "") else [value])
    return "、".join(_csv_text(item) for item in values)


def _csv_paragraphs(value: object) -> str:
    if not isinstance(value, list):
        return ""
    contents = [
        _csv_text(item.get("content"))
        for item in value
        if isinstance(item, dict) and item.get("content") not in (None, "")
    ]
    return "\n\n".join(contents)


def _csv_text(value: object) -> str:
    text = "" if value is None else str(value)
    if text.startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def _relation_split_config_from_form(form: object, config: AppConfig) -> dict[str, object]:
    split_mode = str(_form_value(form, "split_mode", "relation_split_mode") or config.relation_split_mode).strip()
    if split_mode not in RELATION_SPLIT_MODES:
        raise ValueError(f"Unsupported relation split_mode: {split_mode}")

    batch_size = _positive_int(
        _form_value(form, "batch_size", "relation_batch_size"),
        config.relation_batch_size,
        "batch_size",
    )
    chapter_batch_size = _positive_int(
        _form_value(form, "chapter_batch_size", "relation_chapter_batch_size"),
        config.relation_chapter_batch_size,
        "chapter_batch_size",
    )
    paragraph_batch_size = _positive_int(
        _form_value(form, "paragraph_batch_size", "relation_paragraph_batch_size"),
        config.relation_paragraph_batch_size,
        "paragraph_batch_size",
    )
    fixed_section_batch_size = _positive_int(
        _form_value(
            form,
            "fixed_section_batch_size",
            "relation_fixed_section_batch_size",
            "batch_size",
            "relation_batch_size",
        ),
        config.relation_fixed_section_batch_size,
        "fixed_section_batch_size",
    )
    max_batch_tokens = _positive_int(
        _form_value(form, "max_batch_tokens", "relation_max_batch_tokens"),
        config.relation_max_batch_tokens,
        "max_batch_tokens",
    )
    include_parent_title = _bool_value(
        _form_value(form, "include_parent_title", "relation_include_parent_title"),
        config.relation_include_parent_title,
    )

    return {
        "split_mode": split_mode,
        "batch_size": batch_size,
        "chapter_batch_size": chapter_batch_size,
        "paragraph_batch_size": paragraph_batch_size,
        "fixed_section_batch_size": fixed_section_batch_size,
        "max_batch_tokens": max_batch_tokens,
        "include_parent_title": include_parent_title,
    }


def _form_value(form: object, *keys: str) -> str | None:
    for key in keys:
        value = form.get(key) if hasattr(form, "get") else None
        if value not in (None, ""):
            return str(value)
    return None


def _positive_int(value: str | None, default: int, field_name: str) -> int:
    if value is None:
        return max(1, int(default))
    try:
        return max(1, int(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc


def _bool_value(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
