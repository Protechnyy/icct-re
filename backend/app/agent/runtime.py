"""The sole production integration point with LangGraph/LangChain."""
from __future__ import annotations

from copy import deepcopy
import json
import logging
import time
from typing import Any, TypedDict
import warnings

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent

from .budget import Budget, BudgetExhausted, ServiceUnavailable
from .document import DocumentIndex
from .planning import plan_tasks
from .prompts import TASK_TOOLS, execution_prompt
from .trace import Trace, build_result
from .workspace import RelationWorkspace

LOGGER = logging.getLogger(__name__)


class BudgetedModel(BaseChatModel):
    inner: Any
    budget: Any

    @property
    def _llm_type(self):
        return "document-budget-model"

    def bind_tools(self, tools, **kwargs):
        return BudgetedModel(inner=self.inner.bind_tools(tools, **kwargs), budget=self.budget)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.budget.invoke(self.inner, messages)
        return ChatResult(generations=[ChatGeneration(message=response)])


def task_tools(workspace, task, trace, budget):
    identity = task["id"]
    def search_document(query: str) -> dict:
        """按关键词检索当前文档，最多返回5个段落及小节、页码。"""
        return workspace.index.search_document(query)
    def read_section(section_id: str) -> dict:
        """读取当前文档小节，超过4000字会截断并标明。"""
        return workspace.index.read_section(section_id)
    def locate_evidence(quote: str) -> dict:
        """定位原文引文，忽略空白；不命中时给出最接近的句子。"""
        return workspace.index.locate_evidence(quote)
    def query_entity(name: str) -> dict:
        """查询当前文档实体、别名、参与的关系和出现小节。"""
        return workspace.query_entity(name)
    def correct_relation(relation_id: str, replacement_json: str) -> dict:
        """更正关系。replacement_json 为含原文 evidence 和可选 head/relation/tail 的 JSON 对象。"""
        return workspace.correct_relation(relation_id, replacement_json, identity)
    def delete_relation(relation_id: str, reason: str) -> dict:
        """删除不受原文支持的关系，必须给出理由，原内容会存档。"""
        return workspace.delete_relation(relation_id, reason, identity)
    def merge_entities(canonical_name: str, aliases_json: str) -> dict:
        """合并实体。规范名必须在原文出现；aliases_json 是待合并名称的 JSON 数组。"""
        return workspace.merge_entities(canonical_name, aliases_json, identity)
    def add_relation(relation_json: str) -> dict:
        """补充关系。JSON 对象须含 head/relation/tail/evidence，证据必须在原文定位到。"""
        return workspace.add_relation(relation_json, identity)
    functions = {function.__name__: function for function in (
        search_document, read_section, locate_evidence, query_entity, correct_relation,
        delete_relation, merge_entities, add_relation)}
    tools = []
    for name in TASK_TOOLS[task["type"]]:
        function = functions[name]
        schema_tool = StructuredTool.from_function(function)
        def execute(tool_name=name, tool_function=function, **arguments):
            # Tools have no external access, but must still stop writes at the deadline.
            if budget.clock() >= budget.deadline:
                raise BudgetExhausted("核查阶段超时")
            started = time.monotonic()
            try:
                result = tool_function(**arguments)
            except Exception as exc:
                result = {"error": "工具执行失败：" + type(exc).__name__}
            relation_ids = result.get("change", {}).get("relation_ids", [])
            task["relation_ids"] = list(dict.fromkeys([*task.get("relation_ids", []), *relation_ids]))
            trace.emit("tool", identity, result, elapsed_seconds=time.monotonic() - started,
                       tool=tool_name, args=arguments, relation_ids=relation_ids)
            return result
        tools.append(StructuredTool(name=name, description=schema_tool.description,
                                    args_schema=schema_tool.args_schema, func=execute))
    return tools


class VerificationState(TypedDict, total=False):
    workspace: Any
    memory: dict
    trace: Any
    budget: Any
    tasks: list
    cursor: int
    planning: dict
    stopped: str
    result: dict


def run_verification(relations, sections, document_text, config, progress_callback=None, model=None):
    original = deepcopy(relations)
    budget = Budget(config.agent_max_llm_calls, config.agent_timeout_seconds)
    trace = Trace(progress_callback)
    workspace = RelationWorkspace(relations, DocumentIndex(sections, document_text), config.agent_max_added_per_task)
    if config.agent_concurrency > 1:
        LOGGER.warning("AGENT_CONCURRENCY=%s is clamped to 1; verification tasks execute serially", config.agent_concurrency)
    if model is None:
        if not config.agent_api_key:
            raise ValueError("AGENT_API_KEY 未配置")
        model = ChatOpenAI(model=config.agent_model, api_key=config.agent_api_key,
            base_url=config.agent_base_url, timeout=config.agent_timeout_seconds, max_retries=0,
            extra_body={"enable_thinking": False}, callbacks=[],
            max_tokens=2048)
    counted_model = BudgetedModel(inner=model, budget=budget)

    def plan(state):
        started = time.monotonic()
        def call(prompt):
            response = budget.invoke(model, [("system", "只输出核查任务清单 JSON，不输出思考过程。"), ("user", prompt)])
            return response.content
        tasks, metadata = plan_tasks(call, workspace.relations, sections, document_text, config.agent_max_tasks)
        trace.tasks = tasks
        trace.emit("plan", result={"tasks": tasks, "planning": metadata}, elapsed_seconds=time.monotonic() - started)
        return {"tasks": tasks, "cursor": 0, "planning": metadata, "stopped": ""}

    def execute(state):
        task = state["tasks"][state["cursor"]]
        try:
            budget.check()
        except BudgetExhausted as exc:
            return {"stopped": str(exc)}
        task["status"] = "running"
        target = workspace.find(task["target"])
        if target:
            task["relation_ids"] = [target["relation_id"]]
        elif task["type"] == "normalize_entity":
            task["relation_ids"] = [r["relation_id"] for r in workspace.query_entity(task["target"])["relations"]]
        else:
            task["relation_ids"] = [r["relation_id"] for r in workspace.relations
                                    if task["target"] in r.get("source_sections", [])]
        trace.current_task = task
        task_started = time.monotonic()
        trace.emit("task_start", task["id"], task, relation_ids=task["relation_ids"])
        stopped = ""
        try:
            tools = task_tools(workspace, task, trace, budget)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                loop = create_react_agent(counted_model, ToolNode(tools, handle_tool_errors=True),
                                          prompt=execution_prompt(task, workspace))
            last_answer = None
            tool_calls = {}
            for update in loop.stream({"messages": [("user", "请核查这个任务并给出结论。") ]},
                                      {"recursion_limit": config.agent_max_steps}, stream_mode="updates"):
                for payload in update.values():
                    for message in payload.get("messages", []):
                        if isinstance(message, AIMessage):
                            tool_calls.update({call["id"]: call for call in message.tool_calls})
                        if isinstance(message, ToolMessage) and message.status == "error":
                            call = tool_calls.get(message.tool_call_id, {})
                            trace.emit("tool", task["id"], message.content, tool=message.name,
                                       args=call.get("args", {}), error=True)
                        if isinstance(message, AIMessage) and not message.tool_calls:
                            last_answer = message.content if isinstance(message.content, str) else "\n".join(
                                block.get("text", "") for block in message.content
                                if isinstance(block, dict) and block.get("type") == "text")
            if not last_answer or "Sorry, need more steps" in str(last_answer):
                task["status"] = "incomplete"
                task["conclusion"] = "达到工具循环步数上限或模型未给出结论"
            else:
                task["status"] = "completed"
                task["conclusion"] = str(last_answer)
                for relation_id in task["relation_ids"]:
                    relation = workspace.find(relation_id)
                    if relation:
                        current_status = relation["verification"]["status"]
                        workspace.mark(relation, task["id"], "confirmed" if current_status == "unchecked" else current_status)
                related = [r for r in workspace.relations if r["relation_id"] in task["relation_ids"]]
                workspace.conclusions.append({"conclusion": task["conclusion"],
                    "entities": list({r[field] for r in related for field in ("head", "tail")}),
                    "sections": list({str(s) for r in related for s in r.get("source_sections", [])})})
                trace.emit("conclusion", task["id"], task["conclusion"], relation_ids=task["relation_ids"])
        except GraphRecursionError:
            task["status"] = "incomplete"
            task["conclusion"] = "达到工具循环步数上限"
            trace.emit("error", task["id"], task["conclusion"])
        except (BudgetExhausted, ServiceUnavailable) as exc:
            task["status"] = "incomplete"
            task["conclusion"] = str(exc)
            stopped = str(exc)
            trace.emit("error", task["id"], stopped)
        except Exception as exc:
            task["status"] = "failed"
            task["conclusion"] = "任务执行失败：" + type(exc).__name__
            trace.emit("error", task["id"], task["conclusion"])
        trace.current_task = None
        trace.emit("task_end", task["id"], {"status": task["status"], "conclusion": task["conclusion"]},
                   elapsed_seconds=time.monotonic() - task_started, relation_ids=task["relation_ids"])
        return {"cursor": state["cursor"] + 1, "stopped": stopped}

    def route(state):
        return "execute" if state["cursor"] < len(state["tasks"]) and not state["stopped"] else "summarize"

    def summarize(state):
        for task in trace.tasks:
            if task["status"] == "pending":
                task["status"] = "not_executed"
        finished = sum(task["status"] == "completed" for task in trace.tasks)
        any_failure = any(task["status"] != "completed" for task in trace.tasks)
        status = "partial" if any_failure else "completed"
        if state["stopped"] and not finished and "预算" not in state["stopped"] and "超时" not in state["stopped"]:
            status = "failed"
        result = build_result(workspace, trace, budget, state["planning"], status, state["stopped"])
        if state["stopped"] and not finished:
            result["discarded_changes"] = result.pop("changes")
            result["changes"], result["removed_relations"], result["entity_aliases"] = [], [], {}
            for key in ("confirmed_relations", "corrected_relations", "added_relations", "deleted_relations", "entity_merges"):
                result["summary"][key] = 0
            result["summary"]["change_counts"] = {}
        # Final status snapshot includes skipped tasks as well as the completed count.
        if trace.callback:
            trace.emit("conclusion", result={"status": status, "reason": state["stopped"]})
            result["trace"] = deepcopy(trace.events)
        return {"result": result}

    graph = StateGraph(VerificationState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route)
    graph.add_conditional_edges("execute", route)
    graph.add_edge("summarize", END)
    try:
        state = graph.compile().invoke({"workspace": workspace, "trace": trace, "budget": budget,
            "memory": {"entity_aliases": workspace.entity_aliases, "conclusions": workspace.conclusions}},
            {"recursion_limit": config.agent_max_tasks + 4})
    except Exception as exc:
        reason = "核查阶段失败：" + type(exc).__name__
        for task in trace.tasks:
            if task["status"] == "running":
                task["status"] = "failed"
            elif task["status"] == "pending":
                task["status"] = "not_executed"
        trace.current_task = None
        trace.emit("error", result=reason)
        result = build_result(RelationWorkspace(original, workspace.index), trace, budget, {}, "failed", reason)
        result["discarded_changes"] = deepcopy(workspace.changes)
        return original, result
    completed = any(task["status"] == "completed" for task in trace.tasks)
    return (original if state["stopped"] and not completed else deepcopy(workspace.relations)), state["result"]
