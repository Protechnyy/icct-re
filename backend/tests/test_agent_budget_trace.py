import pytest
from types import SimpleNamespace

from app.agent.budget import Budget, BudgetExhausted
from app.agent.trace import Trace, build_result
from test_agent_workspace import make_workspace


def test_retry_counts_against_budget_and_accumulates_usage():
    class RateLimit(Exception):
        status_code = 429
    class Model:
        attempts = 0
        def invoke(self, messages, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RateLimit()
            return SimpleNamespace(usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})
    sleeps = []
    budget = Budget(2, 600, sleep=sleeps.append)
    budget.invoke(Model(), "test")
    assert sleeps == [1] and budget.calls == 2 and budget.tokens["total_tokens"] == 5
    with pytest.raises(BudgetExhausted):
        budget.invoke(Model(), "test")


def test_deadline_prevents_calls_and_discards_late_response():
    current = [0.0]
    budget = Budget(80, 1, clock=lambda: current[0])
    class Model:
        def invoke(self, messages, **kwargs):
            assert 0 < kwargs["timeout"] <= 1
            current[0] = 2
            return SimpleNamespace(usage_metadata={})
    with pytest.raises(BudgetExhausted):
        budget.invoke(Model(), "test")


def test_trace_order_truncation_and_recent_event_bound():
    updates = []
    trace = Trace(updates.append)
    trace.emit("plan", result="任务列表")
    trace.emit("task_start", "t1", "核查")
    trace.emit("tool", "t1", "长" * 600, tool="read_section", args={"section_id": "s1"})
    trace.emit("conclusion", "t1", "结论")
    trace.emit("task_end", "t1", "完成")
    assert [e["kind"] for e in trace.events] == ["plan", "task_start", "tool", "conclusion", "task_end"]
    assert trace.events[2]["truncated"] and len(trace.events[2]["result"]) == 500
    for _ in range(30):
        trace.emit("tool", "t1", "结果")
    assert len(updates[-1]["recent_events"]) == 20
    assert [e["seq"] for e in trace.events] == list(range(1, 36))


def test_summary_counts_actual_changes():
    workspace, _ = make_workspace()
    workspace.delete_relation(workspace.relations[0]["relation_id"], "无依据", "t1")
    trace = Trace()
    trace.tasks = [{"id": "t1", "status": "completed"}]
    result = build_result(workspace, trace, Budget(80, 600), {}, "completed")
    assert result["summary"]["deleted_relations"] == len(result["removed_relations"]) == 1
    assert result["summary"]["task_counts"] == {"completed": 1}
    assert result["summary"]["change_counts"] == {"deletion": 1}
