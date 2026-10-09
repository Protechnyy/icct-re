from collections import Counter
from copy import deepcopy
import json
from threading import RLock

from ..utils import utcnow_iso


def event_preview(event):
    return deepcopy({key: value for key, value in event.items() if key != "result_data"})


def result_summary(result):
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if result.get("error"):
            return str(result["error"])
        if "matches" in result:
            return f"检索到 {len(result['matches'])} 条原文" if result["matches"] else "未找到相关原文"
        if "matched" in result:
            return f"已定位证据，共 {len(result.get('locations', []))} 处" if result["matched"] else "证据未能定位"
        if "change" in result:
            return "已执行关系改动：" + str(result["change"]["type"])
        if "text" in result:
            return "已读取小节：" + str(result.get("title") or result.get("section_id", ""))
        if "found" in result:
            return f"已查询实体：{result.get('name', '')}，关联 {len(result.get('relations', []))} 条关系"
        if "tasks" in result:
            return f"已规划 {len(result['tasks'])} 项核查任务"
        if "conclusion" in result:
            return str(result["conclusion"])
    return json.dumps(result, ensure_ascii=False)


class Trace:
    def __init__(self, callback=None, *, document_task_id=None, event_callback=None):
        self.events = []
        self.tasks = []
        self.callback = callback
        self.current_task = None
        self.document_task_id = document_task_id
        self.event_callback = event_callback
        self.phase = "preparing"
        self.status = "running"
        self.lock = RLock()
        self.tool_calls = {}
        self.finished_calls = set()

    def emit(self, kind, task_id=None, result=None, elapsed_seconds=0.0, **details):
        with self.lock:
            if kind == "plan_start":
                self.phase = "planning"
            elif kind in ("plan", "task_start"):
                self.phase = "executing"
            elif kind == "phase_end":
                self.phase = "finished"
                self.status = result["status"]
            encoded = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            summary = details.pop("summary", result_summary(result))
            status = details.pop("status", "running" if kind.endswith("_start") else "completed")
            event = {"version": 2, "document_task_id": self.document_task_id,
                     "seq": len(self.events) + 1, "task_id": task_id, "kind": kind,
                     "timestamp": utcnow_iso(), "phase": self.phase, "status": status,
                     "result": encoded[:500], "summary": summary[:500],
                     "truncated": len(encoded) > 500, "result_data": deepcopy(result),
                     "detail_available": result is not None,
                     "elapsed_seconds": round(elapsed_seconds, 4), **deepcopy(details)}
            if self.event_callback:
                self.event_callback(event)
            self.events.append(event)
            progress = self.progress()
            if self.callback:
                self.callback(progress)
            return event

    def progress(self):
        return {"version": 2, "phase": self.phase, "status": self.status,
                "last_seq": len(self.events), "total_tasks": len(self.tasks),
                "completed_tasks": sum(task["status"] == "completed" for task in self.tasks),
                "finished_tasks": sum(task["status"] not in ("pending", "running") for task in self.tasks),
                "tasks": deepcopy(self.tasks), "current_task": deepcopy(self.current_task),
                "recent_events": [event_preview(event) for event in self.events[-20:]]}

    def finish(self, result):
        accepted = [change["change_id"] for change in result.get("changes", [])]
        discarded = [change["change_id"] for change in result.get("discarded_changes", [])]
        self.current_task = None
        self.emit("phase_end", result={"status": result["status"], "reason": result.get("reason", ""),
                  "summary": result.get("summary", {}), "accepted_change_ids": accepted,
                  "discarded_change_ids": discarded}, status=result["status"])
        result.update({"version": 2, "last_seq": len(self.events), "trace": deepcopy(self.events),
                       "tasks": deepcopy(self.tasks), "accepted_change_ids": accepted,
                       "discarded_change_ids": discarded})
        return result


def build_result(workspace, trace, budget, planning, status, reason=""):
    task_counts = Counter(task["status"] for task in trace.tasks)
    states = Counter(item["verification"]["status"] for item in workspace.relations)
    operations = Counter(change["type"] for change in workspace.changes)
    return {"version": 2, "last_seq": len(trace.events), "status": status, "reason": reason, "planning": planning,
            "summary": {"total_tasks": len(trace.tasks), "task_counts": dict(task_counts),
                        "confirmed_relations": states["confirmed"], "corrected_relations": states["corrected"],
                        "added_relations": states["added"], "deleted_relations": len(workspace.removed_relations),
                        "entity_merges": operations["entity_merge"], "change_counts": dict(operations),
                        "llm_calls": budget.calls, "tokens": dict(budget.tokens),
                        "elapsed_seconds": budget.elapsed_seconds},
            "tasks": deepcopy(trace.tasks), "trace": deepcopy(trace.events),
            "removed_relations": deepcopy(workspace.removed_relations),
            "entity_aliases": deepcopy(workspace.entity_aliases), "changes": deepcopy(workspace.changes)}
