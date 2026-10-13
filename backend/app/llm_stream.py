from __future__ import annotations

import json
from typing import Any

from redis import Redis

# 单个任务的事件通道容量上限：XADD 后按近似值裁剪最旧事件，
# 任务终态后不删除，供前端完成后回看。
STREAM_MAXLEN = 2000
STAGE_STREAM_ID = "pipeline"


def stream_key(task_id: str) -> str:
    return f"task:{task_id}:llm"


def publish_delta(redis: Redis, task_id: str, stream_id: str, kind: str, text: str) -> str:
    return redis.xadd(
        stream_key(task_id),
        {"stream_id": stream_id, "kind": kind, "text": text},
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )


def publish_stage(redis: Redis, task_id: str, payload: dict[str, Any]) -> str:
    return redis.xadd(
        stream_key(task_id),
        {"stream_id": STAGE_STREAM_ID, "kind": "stage", "text": json.dumps(payload, ensure_ascii=False)},
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )


def publish_done(redis: Redis, task_id: str, status: str) -> str:
    return redis.xadd(
        stream_key(task_id),
        {"stream_id": STAGE_STREAM_ID, "kind": "done", "text": status},
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )


def decode_entry(entry_id: str, fields: dict[str, Any]) -> dict[str, str]:
    return {
        "seq": entry_id,
        "stream_id": str(fields.get("stream_id", "")),
        "kind": str(fields.get("kind", "")),
        "text": str(fields.get("text", "")),
    }
