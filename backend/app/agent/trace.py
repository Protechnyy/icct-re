"""Serializable events; no model reasoning is persisted."""
from collections import Counter
from copy import deepcopy
import json
import logging

LOGGER = logging.getLogger(__name__)


class Trace:
    def __init__(self, callback=None):
        self.events = []
        self.tasks = []
        self.callback = callback
        self.current_task = None

    def emit(self, kind, task_id=None, result=None, elapsed_seconds=0.0, **details):
        encoded = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        event = {"seq": len(self.events) + 1, "task_id": task_id, "kind": kind,
                 "result": encoded[:500], "truncated": len(encoded) > 500,
                 "elapsed_seconds": round(elapsed_seconds, 4), **deepcopy(details)}
        self.events.append(event)
        if self.callback:
            progress = {"total_tasks": len(self.tasks),
                        "completed_tasks": sum(task["status"] == "completed" for task in self.tasks),
                        "finished_tasks": sum(task["status"] not in ("pending", "running") for task in self.tasks),
                        "tasks": deepcopy(self.tasks), "current_task": deepcopy(self.current_task),
                        "recent_events": deepcopy(self.events[-20:])}
            try:
                self.callback(progress)
            except Exception:
                LOGGER.exception("Unable to report verification progress")
        return event


def build_result(workspace, trace, budget, planning, status, reason=""):
    task_counts = Counter(task["status"] for task in trace.tasks)
    states = Counter(item["verification"]["status"] for item in workspace.relations)
    operations = Counter(change["type"] for change in workspace.changes)
    return {"status": status, "reason": reason, "planning": planning,
            "summary": {"total_tasks": len(trace.tasks), "task_counts": dict(task_counts),
                        "confirmed_relations": states["confirmed"], "corrected_relations": states["corrected"],
                        "added_relations": states["added"], "deleted_relations": len(workspace.removed_relations),
                        "entity_merges": operations["entity_merge"], "change_counts": dict(operations),
                        "llm_calls": budget.calls, "tokens": dict(budget.tokens),
                        "elapsed_seconds": budget.elapsed_seconds},
            "tasks": deepcopy(trace.tasks), "trace": deepcopy(trace.events),
            "removed_relations": deepcopy(workspace.removed_relations),
            "entity_aliases": deepcopy(workspace.entity_aliases), "changes": deepcopy(workspace.changes)}
