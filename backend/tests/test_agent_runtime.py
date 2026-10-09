from copy import deepcopy
from dataclasses import replace
import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import verify_document
from app.agent.budget import Budget
from app.agent.prompts import TASK_TOOLS, execution_prompt
from app.agent.runtime import task_tools
from app.agent.trace import Trace
from test_agent_workspace import make_workspace
from test_pipeline import build_config


class ScriptedModel(BaseChatModel):
    replies: list[Any]
    calls: list[Any] = []
    position: int = 0

    @property
    def _llm_type(self):
        return "scripted-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        reply = self.replies[self.position]
        self.position += 1
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            reply = AIMessage(content=reply)
        reply.usage_metadata = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call_tool(name, arguments):
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": "call-test", "type": "tool_call"}])


def configuration(tmp_path, **kwargs):
    return replace(build_config(tmp_path), agent_enabled=True, **kwargs)


def plan(*tasks):
    return json.dumps({"tasks": [{"type": kind, "target": target, "reason": "测试"} for kind, target in tasks]})


def test_tool_sets_and_task_prompts_match_design():
    workspace, _ = make_workspace()
    expected = {
        "verify_evidence": {"search_document", "read_section", "locate_evidence", "correct_relation", "delete_relation"},
        "normalize_entity": {"query_entity", "search_document", "read_section", "merge_entities"},
        "check_section": {"read_section", "query_entity", "locate_evidence", "add_relation"}}
    for kind, names in expected.items():
        task = {"id": "t1", "type": kind, "target": workspace.relations[0]["relation_id"]}
        tools = task_tools(workspace, task, Trace(), Budget(80, 600))
        assert {t.name for t in tools} == names == set(TASK_TOOLS[kind]) and len(tools) <= 5
        assert "任务" in execution_prompt(task, workspace)


def test_two_tasks_apply_expected_changes_and_trace(tmp_path):
    workspace, original = make_workspace()
    first, second = [r["relation_id"] for r in workspace.relations]
    model = ScriptedModel(replies=[plan(("verify_evidence", first), ("verify_evidence", second)),
        call_tool("correct_relation", {"relation_id": first, "replacement_json": json.dumps({"head": "第三营", "evidence": "第三营部署于河谷。"})}),
        "已根据小节 s1 第3页更正指代。", call_tool("delete_relation", {"relation_id": second, "reason": "与更正后的关系重复"}),
        "已删除重复关系。"])
    output, result = verify_document(original, list(workspace.index.sections.values()), workspace.index.document_text,
                                     configuration(tmp_path), model=model)
    assert result["status"] == "completed" and len(output) == 1
    assert output[0]["head"] == "第三营" and output[0]["verification"]["status"] == "corrected"
    assert len(result["removed_relations"]) == 1
    assert result["summary"]["llm_calls"] == 5 and result["summary"]["tokens"]["total_tokens"] == 15
    assert [e["kind"] for e in result["trace"]] == ["plan", "task_start", "tool", "conclusion", "task_end",
                                                  "task_start", "tool", "conclusion", "task_end"]
    assert "relation_id" not in original[0]


def test_later_task_prompt_contains_alias_memory(tmp_path):
    workspace, original = make_workspace()
    first = workspace.relations[0]["relation_id"]
    model = ScriptedModel(replies=[plan(("normalize_entity", "该营"), ("verify_evidence", first)),
        call_tool("merge_entities", {"canonical_name": "第三营", "aliases_json": '["该营"]'}),
        "该营即第三营。", "原文证据支持该关系。"])
    output, result = verify_document(original, list(workspace.index.sections.values()), workspace.index.document_text,
                                     configuration(tmp_path), model=model)
    assert result["status"] == "completed"
    system = str(model.calls[-1][0].content)
    assert "别名：第三营 = 该营" in system
    assert result["entity_aliases"] == {"第三营": ["该营"]}


def test_budget_preserves_completed_tasks_and_skips_remaining(tmp_path):
    workspace, original = make_workspace()
    first, second = [r["relation_id"] for r in workspace.relations]
    model = ScriptedModel(replies=[plan(("verify_evidence", first), ("verify_evidence", second)),
        call_tool("correct_relation", {"relation_id": first, "replacement_json": '{"head":"第三营","evidence":"第三营部署于河谷。"}'}),
        "已更正。"])
    output, result = verify_document(original, list(workspace.index.sections.values()), workspace.index.document_text,
                                     configuration(tmp_path, agent_max_llm_calls=3), model=model)
    assert result["status"] == "partial" and output[0]["head"] == "第三营"
    assert [t["status"] for t in result["tasks"]] == ["completed", "not_executed"]


def test_tool_rejection_does_not_modify_relation_or_fail_task(tmp_path):
    workspace, original = make_workspace()
    first = workspace.relations[0]["relation_id"]
    model = ScriptedModel(replies=[plan(("verify_evidence", first)),
        call_tool("correct_relation", {"relation_id": first, "replacement_json": '{"evidence":"虚构"}'}), "保留原关系。"])
    output, result = verify_document(original, list(workspace.index.sections.values()), workspace.index.document_text,
                                     configuration(tmp_path), model=model)
    assert result["status"] == "completed" and output[0]["head"] == "该营"
    assert "证据无法" in next(e for e in result["trace"] if e["kind"] == "tool")["result"]


def test_task_failure_continues_next_task(tmp_path):
    workspace, original = make_workspace()
    first, second = [r["relation_id"] for r in workspace.relations]
    model = ScriptedModel(replies=[plan(("verify_evidence", first), ("verify_evidence", second)),
                                  RuntimeError("task failure"), "第二条关系得到原文支持。"])
    output, result = verify_document(original, list(workspace.index.sections.values()), workspace.index.document_text,
                                     configuration(tmp_path), model=model)
    assert result["status"] == "partial"
    assert [t["status"] for t in result["tasks"]] == ["failed", "completed"]


def test_stage_exception_returns_original_relations(tmp_path, monkeypatch):
    workspace, original = make_workspace()
    def fail(*args):
        raise RuntimeError("internal failure")
    monkeypatch.setattr("app.agent.runtime.run_verification", fail)
    output, result = verify_document(original, [], workspace.index.document_text, configuration(tmp_path))
    assert output == original and result["status"] == "failed"


def test_service_unavailable_before_completed_task_reverts_changes(tmp_path):
    workspace, original = make_workspace()
    first = workspace.relations[0]["relation_id"]
    class Offline(Exception):
        status_code = 503
    model = ScriptedModel(replies=[plan(("verify_evidence", first)),
        call_tool("delete_relation", {"relation_id": first, "reason": "原文不支持"}), Offline()])
    output, result = verify_document(original, list(workspace.index.sections.values()), workspace.index.document_text,
                                     configuration(tmp_path), model=model)
    assert output == original and result["status"] == "failed"
    assert not result["changes"] and not result["removed_relations"] and result["discarded_changes"]


def test_step_limit_marks_incomplete_and_keeps_valid_edit(tmp_path):
    workspace, original = make_workspace()
    first = workspace.relations[0]["relation_id"]
    model = ScriptedModel(replies=[plan(("verify_evidence", first)),
        call_tool("correct_relation", {"relation_id": first, "replacement_json": '{"head":"第三营","evidence":"第三营部署于河谷。"}'}),
        call_tool("read_section", {"section_id": "s1"})])
    output, result = verify_document(original, list(workspace.index.sections.values()), workspace.index.document_text,
                                     configuration(tmp_path, agent_max_steps=3), model=model)
    assert result["status"] == "partial" and result["tasks"][0]["status"] == "incomplete"
    assert output[0]["head"] == "第三营"


def test_progress_planning_running_and_finished(tmp_path):
    workspace, original = make_workspace()
    first = workspace.relations[0]["relation_id"]
    model = ScriptedModel(replies=[plan(("verify_evidence", first)),
        call_tool("read_section", {"section_id": "s1"}), "证据正确。"])
    updates = []
    verify_document(original, list(workspace.index.sections.values()), workspace.index.document_text,
                    configuration(tmp_path), updates.append, model=model)
    assert updates[0]["total_tasks"] == 1 and updates[0]["tasks"][0]["status"] == "pending"
    assert any(u["current_task"] and u["recent_events"][-1]["kind"] == "tool" for u in updates)
    assert updates[-1]["completed_tasks"] == 1 and updates[-1]["current_task"] is None


def test_unknown_tool_and_bad_arguments_are_returned_and_traced(tmp_path):
    workspace, original = make_workspace()
    first = workspace.relations[0]["relation_id"]
    model = ScriptedModel(replies=[plan(("verify_evidence", first)),
        call_tool("invented_tool", {"query": "测试"}), call_tool("read_section", {}), "保持关系不变。"])
    output, result = verify_document(original, list(workspace.index.sections.values()), workspace.index.document_text,
                                     configuration(tmp_path), model=model)
    assert result["status"] == "completed"
    failures = [event for event in result["trace"] if event.get("error")]
    assert [event["tool"] for event in failures] == ["invented_tool", "read_section"]
    assert output[0]["head"] == original[0]["head"]


def test_unexpected_graph_failure_preserves_trace_and_costs(tmp_path, monkeypatch):
    workspace, original = make_workspace()
    first = workspace.relations[0]["relation_id"]
    model = ScriptedModel(replies=[plan(("verify_evidence", first)), "有原文依据。"])
    def unexpected(*args):
        raise RuntimeError("summary crash")
    monkeypatch.setattr("app.agent.runtime.execution_prompt", unexpected)
    # Prompt construction is contained at the task boundary and still records the failure.
    output, result = verify_document(original, list(workspace.index.sections.values()), workspace.index.document_text,
                                     configuration(tmp_path), model=model)
    assert result["status"] == "partial"
    assert result["summary"]["llm_calls"] == 1
    assert result["trace"][-1]["kind"] == "task_end"
