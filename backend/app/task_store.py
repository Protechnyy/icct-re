from __future__ import annotations

import json
from typing import Any

from redis import Redis

from .types import TaskStatus
from .utils import utcnow_iso


TASK_QUEUE_KEY = "docre:tasks:queue"


class RedisTaskStore:
    def __init__(self, redis_url: str) -> None:
        self.redis = Redis.from_url(redis_url, decode_responses=True)

    def _task_key(self, task_id: str) -> str:
        return f"docre:task:{task_id}"

    def _result_key(self, task_id: str) -> str:
        return f"docre:result:{task_id}"

    def healthcheck(self) -> bool:
        return bool(self.redis.ping())

    def create_task(self, task: TaskStatus, payload: dict[str, Any]) -> None:
        data = task.to_dict()
        data["payload"] = payload
        self.redis.set(self._task_key(task.task_id), json.dumps(data, ensure_ascii=False))

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        value = self.redis.get(self._task_key(task_id))
        return json.loads(value) if value else None

    def enqueue_task(self, task_id: str) -> None:
        self.redis.rpush(TASK_QUEUE_KEY, task_id)

    def dequeue_task(self, timeout: int) -> str | None:
        item = self.redis.blpop(TASK_QUEUE_KEY, timeout=timeout)
        if not item:
            return None
        _, task_id = item
        return task_id

    def update_task(self, task_id: str, **updates: Any) -> dict[str, Any]:
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")
        task.update(updates)
        task["updated_at"] = utcnow_iso()
        self.redis.set(self._task_key(task_id), json.dumps(task, ensure_ascii=False))
        return task

    def set_result(self, task_id: str, result: dict[str, Any]) -> None:
        self.redis.set(self._result_key(task_id), json.dumps(result, ensure_ascii=False))
        self.update_task(task_id, result_ready=True)

    def get_result(self, task_id: str) -> dict[str, Any] | None:
        value = self.redis.get(self._result_key(task_id))
        return json.loads(value) if value else None

