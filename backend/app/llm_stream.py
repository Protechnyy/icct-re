from __future__ import annotations

import json
from typing import Any

from redis import Redis

# 每个任务的流式事件完整保留，供前端随时重连或完成后回看。事件粒度是模型返回的
# 每个增量片段，单个任务的事件数量与文档规模成正比且没有固定上限，按条数裁剪会把
# 早期批次和每条流的开头文本裁掉，重放时表现为输出缺少内容，因此不做容量裁剪。
STAGE_STREAM_ID = "pipeline"


def stream_key(task_id: str) -> str:
    return f"task:{task_id}:llm"


def publish_delta(redis: Redis, task_id: str, stream_id: str, kind: str, text: str) -> str:
    return redis.xadd(
        stream_key(task_id),
        {"stream_id": stream_id, "kind": kind, "text": text},
    )


def publish_stage(redis: Redis, task_id: str, payload: dict[str, Any]) -> str:
    return redis.xadd(
        stream_key(task_id),
        {"stream_id": STAGE_STREAM_ID, "kind": "stage", "text": json.dumps(payload, ensure_ascii=False)},
    )


def publish_done(redis: Redis, task_id: str, status: str) -> str:
    return redis.xadd(
        stream_key(task_id),
        {"stream_id": STAGE_STREAM_ID, "kind": "done", "text": status},
    )


def decode_entry(entry_id: str, fields: dict[str, Any]) -> dict[str, str]:
    return {
        "seq": entry_id,
        "stream_id": str(fields.get("stream_id", "")),
        "kind": str(fields.get("kind", "")),
        "text": str(fields.get("text", "")),
    }
