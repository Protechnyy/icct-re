from __future__ import annotations

import json
from typing import Any

from redis import Redis

from .types import TaskStatus
from .utils import utcnow_iso
from .agent.trace import event_preview


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

    def _agent_key(self, task_id: str, part: str) -> str:
        return f"docre:agent:{task_id}:{part}"

    def append_agent_event(self, task_id: str, event: dict[str, Any]) -> None:
        if event["document_task_id"] != task_id:
            raise ValueError("事件的文档任务编号不一致")
        events_key = self._agent_key(task_id, "events")
        if event["seq"] != self.redis.llen(events_key) + 1:
            raise ValueError("事件序号必须连续递增")
        meta = {"version": event["version"], "last_seq": event["seq"], "phase": event["phase"]}
        if event["kind"] == "phase_start":
            meta["status"] = "running"
        if event["kind"] == "plan":
            meta["planning"] = json.dumps(event["result_data"]["planning"], ensure_ascii=False)
        if event["kind"] == "phase_end":
            meta["status"] = event["result_data"]["status"]
            meta["final"] = json.dumps(event["result_data"], ensure_ascii=False)
        with self.redis.pipeline(transaction=True) as transaction:
            transaction.rpush(events_key, json.dumps(event_preview(event), ensure_ascii=False))
            transaction.hset(self._agent_key(task_id, "details"), str(event["seq"]), json.dumps(event, ensure_ascii=False))
            transaction.hset(self._agent_key(task_id, "meta"), mapping=meta)
            transaction.execute()

    def get_agent_events(self, task_id: str, after_seq: int = 0, limit: int = 100) -> dict[str, Any] | None:
        task = self.get_task(task_id)
        if task is None:
            return None
        meta = self.redis.hgetall(self._agent_key(task_id, "meta"))
        progress = task.get("agent_progress", {})
        if meta:
            last_seq = int(meta["last_seq"])
            stop = min(after_seq + limit, last_seq)
            values = self.redis.lrange(self._agent_key(task_id, "events"), after_seq, stop - 1) if stop > after_seq else []
            events = [json.loads(value) for value in values]
            final = json.loads(meta.get("final", "{}"))
            state = {"version": int(meta["version"]), "phase": meta["phase"],
                     "status": meta.get("status", "running"), "tasks": progress.get("tasks", []),
                     "current_task": progress.get("current_task"),
                     "planning": json.loads(meta.get("planning", "{}")), **final}
        else:
            result = self.get_result(task_id) or {}
            agent = result.get("agent_result", {})
            trace = agent.get("trace", [])
            events = [{**event_preview(event), "detail_available": "result_data" in event,
                       "summary": event.get("summary", event.get("result", ""))}
                      for event in trace if event["seq"] > after_seq][:limit]
            last_seq = trace[-1]["seq"] if trace else 0
            state = {"version": agent.get("version", 1), "phase": progress.get("phase", "finished" if agent else "pending"),
                     "status": agent.get("status", progress.get("status", "pending")),
                     "tasks": agent.get("tasks", progress.get("tasks", [])),
                     "current_task": progress.get("current_task"), "planning": agent.get("planning", {}),
                     "summary": agent.get("summary", {}), "reason": agent.get("reason", ""),
                     "accepted_change_ids": agent.get("accepted_change_ids", []),
                     "discarded_change_ids": agent.get("discarded_change_ids", [])}
        next_seq = events[-1]["seq"] if events else after_seq
        return {**state, "document_task_id": task_id, "events": events, "next_seq": next_seq,
                "last_seq": last_seq, "has_more": next_seq < last_seq}

    def get_agent_event(self, task_id: str, seq: int) -> dict[str, Any] | None:
        value = self.redis.hget(self._agent_key(task_id, "details"), str(seq))
        if value:
            return json.loads(value)
        result = self.get_result(task_id) or {}
        event = next((event for event in result.get("agent_result", {}).get("trace", []) if event["seq"] == seq), None)
        if event is None:
            return None
        return {**event, "detail_available": "result_data" in event}

