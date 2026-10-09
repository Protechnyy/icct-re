import Button from "@arco-design/web-react/es/Button";
import Alert from "@arco-design/web-react/es/Alert";
import Card from "@arco-design/web-react/es/Card";
import Drawer from "@arco-design/web-react/es/Drawer";
import Empty from "@arco-design/web-react/es/Empty";
import Tag from "@arco-design/web-react/es/Tag";
import Typography from "@arco-design/web-react/es/Typography";
import { IconRobot, IconLoading, IconCloseCircle, IconRight, IconDown, IconSearch, IconFile, IconLocation, IconTool } from "@arco-design/web-react/icon";
import { useEffect, useMemo, useRef, useState } from "react";
import "@arco-design/web-react/es/Drawer/style/css.js";

export const VERIFICATION_LABELS = { unchecked: "未核查", confirmed: "已确认", corrected: "已更正", added: "新增" };
const TASK_LABELS = { verify_evidence: "核查证据", normalize_entity: "归一实体", check_section: "检查小节漏抽" };
const TOOL_LABELS = { search_document: "检索原文", read_section: "读取小节", locate_evidence: "定位证据", query_entity: "查询实体", correct_relation: "更正关系", delete_relation: "删除关系", merge_entities: "合并实体", add_relation: "补充关系" };
const TASK_STATUS_LABELS = { pending: "待执行", running: "进行中", completed: "已完成", incomplete: "未完成", failed: "失败", not_executed: "未执行" };

const CHANGE_LABELS = { correction: "更正关系", deletion: "删除关系", addition: "补充关系", entity_merge: "合并实体" };
const RECENT_STEP_LIMIT = 5;

function executionSteps(events) {
  const steps = new Map();
  for (const event of events) {
    if (!["model_start", "model_end", "tool_start", "tool"].includes(event.kind)) continue;
    const key = event.call_id || `event-${event.seq}`;
    const previous = steps.get(key);
    steps.set(key, { ...previous, ...event, key,
      firstSeq: previous?.firstSeq || event.seq,
      status: event.status || (event.error ? "error" : "completed"),
      model: event.kind.startsWith("model_") });
  }
  return [...steps.values()].sort((first, second) => first.firstSeq - second.firstSeq);
}

function EvidenceContent({ data }) {
  if (!data || typeof data !== "object") return null;
  const matches = data.matches || data.locations || [];
  const candidates = data.closest_sentences || [];
  return <div className="agent-evidence">
    {data.error && <p className="agent-error-text">{data.error}</p>}
    {data.matched === false && <p>证据未能定位</p>}
    {data.matches?.length === 0 && <p>未找到相关原文</p>}
    {[...matches.map((item) => ({ ...item, candidate: false })), ...candidates.map((item) => ({ ...item, candidate: true }))].map((item, index) => <blockquote key={index}>
      <div className="agent-source-line"><Tag>{item.candidate ? "候选原文" : data.matched ? "已定位证据" : "命中原文"}</Tag>
        <span>小节：{item.section_id ?? "未提供"}</span><span>页码：{item.page ?? "未提供"}</span></div>
      <p>{item.content}</p>
    </blockquote>)}
    {data.text !== undefined && <blockquote><div className="agent-source-line"><span>小节：{data.title || data.section_id}</span>
      <span>页码：{data.page_start ?? "未提供"}{data.page_end !== undefined && data.page_end !== data.page_start ? `–${data.page_end}` : ""}</span></div>
      <p>{data.text}</p>{data.truncated && <p>工具返回的原文已截断</p>}</blockquote>}
    {data.found !== undefined && <div><p>实体：{data.name} · {data.found ? "已找到" : "未找到"}</p>
      {data.aliases?.length > 0 && <p>别名：{data.aliases.join("、")}</p>}
      {data.relations?.length > 0 && <RelationSnapshot title="关联关系" relations={data.relations} />}</div>}
  </div>;
}

function ChangeContent({ change, adopted, terminal }) {
  return <div className="agent-operation">
    <div className="agent-source-line"><strong>{CHANGE_LABELS[change.type] || change.type}</strong>
      <Tag color={terminal ? adopted ? "green" : "orange" : "gray"}>{terminal ? adopted ? "已被最终结果采用" : "未被最终结果采用" : "已执行，最终采用情况待确认"}</Tag></div>
    {change.canonical_name && <p>规范名称：{change.canonical_name}；合并名称：{(change.aliases || []).join("、")}</p>}
    <RelationSnapshot title="改动前" relations={change.before || []} />
    <RelationSnapshot title="改动后" relations={change.after || []} />
    <p>原文依据：{change.evidence || change.reason || "未提供"}</p>
    {change.reason && <p>理由：{change.reason}</p>}
    <p>页码：{change.pages?.join("、") || "未提供"}</p>
  </div>;
}

export function AgentWorkspace({ task, agent, stream = {} }) {
  const documentId = task.task_id;
  const [views, setViews] = useState({});
  const executionPaneRef = useRef(null);
  const progress = task.agent_progress || {};
  const events = stream.events?.length ? stream.events : agent?.trace || progress.recent_events || [];
  const status = stream.status || agent?.status || progress.status || "running";
  const terminal = ["completed", "partial", "failed", "skipped", "disabled"].includes(status);
  const view = views[documentId] || { expanded: {}, collapsed: terminal };
  const updateView = (updates) => setViews((current) => ({ ...current, [documentId]: { ...(current[documentId] || view), ...updates } }));
  const steps = useMemo(() => executionSteps(events), [events]);
  const visibleSteps = view.showHistory ? steps : steps.slice(-RECENT_STEP_LIMIT);
  const failedCount = steps.filter((step) => ["error", "rejected"].includes(step.status)).length;
  const executionPaneId = `agent-execution-${documentId}`;
  const accepted = new Set(stream.accepted_change_ids || agent?.accepted_change_ids || []);
  useEffect(() => {
    if (executionPaneRef.current) executionPaneRef.current.scrollTop = 0;
  }, [documentId, view.showHistory]);
  useEffect(() => {
    for (const step of steps) {
      if (view.expanded[step.key] && step.detail_available && !stream.details?.[step.seq]) {
        stream.readDetail(step.seq);
      }
    }
  }, [steps, view.expanded, stream.details, stream.readDetail]);

  if (status === "disabled" || (!agent && !stream.eligible && !task.agent_progress && task.stage !== "agent_verification")) return null;

  function toggleStep(step) {
    const expanded = !view.expanded[step.key];
    updateView({ expanded: { ...view.expanded, [step.key]: expanded } });
    if (expanded && step.detail_available && !stream.details?.[step.seq]) stream.readDetail(step.seq);
  }

  function showCurrentStep() {
    updateView({ showHistory: false });
    requestAnimationFrame(() => {
      const pane = executionPaneRef.current;
      if (pane) pane.scrollTop = pane.scrollHeight;
    });
  }

  return <section className="agent-workspace" aria-label="智能体校验过程" data-document-id={documentId}>
    <header className="agent-workspace-header">
      <div className="agent-workspace-heading"><IconRobot /><h3>智能体校验过程</h3>
        <span className="agent-workspace-state" role="status">{status === "completed" ? "校验完成" : status === "failed" ? "校验失败" : terminal ? "校验已结束" : "正在核查"}</span>
        {steps.length > 0 && <span className="agent-step-count">共 {steps.length} 个步骤</span>}
        {failedCount > 0 && <span className="agent-step-failures">{failedCount} 个步骤未成功</span>}</div>
      <div className="agent-workspace-controls">
        {!view.collapsed && !terminal && steps.length > 0 && <Button type="text" size="mini" onClick={showCurrentStep}>查看当前步骤</Button>}
        <Button type="text" size="mini" onClick={() => updateView({ collapsed: !view.collapsed })} aria-expanded={!view.collapsed}>{view.collapsed ? "展开过程" : "折叠过程"}</Button></div>
    </header>
    {stream.error && <Alert type="error" title="读取执行过程失败" content={stream.error} action={<Button size="mini" onClick={stream.retry}>重新读取</Button>} />}
    {!view.collapsed && <div className="agent-workspace-body">
      {steps.length > RECENT_STEP_LIMIT && <div className="agent-history-controls">
        <span>{view.showHistory ? "显示全部步骤" : `显示最近 ${RECENT_STEP_LIMIT} 个步骤`}</span>
        <Button type="text" size="mini" onClick={() => updateView({ showHistory: !view.showHistory })} aria-expanded={Boolean(view.showHistory)} aria-controls={executionPaneId}>
          {view.showHistory ? "只看最近步骤" : `查看全部 ${steps.length} 个步骤`}
        </Button>
      </div>}
      <div className="agent-execution-pane" id={executionPaneId} ref={executionPaneRef} role="region" tabIndex={0} aria-label="校验执行步骤">
        <ol className="agent-timeline">{visibleSteps.map((step) => {
          const detail = stream.details?.[step.seq];
          const data = detail?.event?.result_data ?? step.result_data;
          const action = step.model ? "调用模型" : TOOL_LABELS[step.tool] || "执行操作";
          const running = step.status === "running" && !terminal;
          const failed = ["error", "rejected"].includes(step.status);
          const context = step.model ? "" : step.args?.query || step.args?.section_id || step.args?.name || "";
          const label = `${running ? "正在" : failed ? "未能" : step.status === "running" ? "未完成" : "已"}${action}`;
          const expandable = !step.model && Boolean(step.detail_available || data);
          const StepIcon = step.model ? IconRobot : step.tool === "search_document" ? IconSearch : step.tool === "read_section" ? IconFile : step.tool === "locate_evidence" ? IconLocation : IconTool;
          return <li key={step.key} id={`agent-step-${documentId}-${step.key}`} data-call-id={step.key}>
            <span className={`agent-step-icon ${running ? "is-running" : failed ? "is-error" : ""}`} aria-hidden="true">{running ? <IconLoading spin /> : failed ? <IconCloseCircle /> : <StepIcon />}</span>
            <div className="agent-step-content">
              {expandable ? <button className="agent-step-toggle" onClick={() => toggleStep(step)} aria-expanded={Boolean(view.expanded[step.key])} aria-controls={`agent-detail-${documentId}-${step.key}`}>
                <span className="agent-step-label">{label}{context && <span className="agent-step-context">：{String(context)}</span>}</span>
                {view.expanded[step.key] ? <IconDown aria-hidden="true" /> : <IconRight aria-hidden="true" />}
              </button> : <div className="agent-step-label">{label}{context && <span className="agent-step-context">：{String(context)}</span>}</div>}
              {failed && <p className="agent-error-text">{step.summary || "调用失败"}</p>}
              {expandable && <div id={`agent-detail-${documentId}-${step.key}`} hidden={!view.expanded[step.key]} className="agent-step-detail">
                {view.expanded[step.key] && <>
                {detail?.loading ? <p role="status">正在读取详情…</p> : detail?.error ? <div><p className="agent-error-text">读取详情失败：{detail.error}</p><Button size="mini" onClick={() => stream.readDetail(step.seq)}>重新读取详情</Button></div> : <>
                  {data ? <EvidenceContent data={data} /> : <p>{step.status === "running" ? "等待调用结果" : "结构化详情未记录"}</p>}
                  {data?.change && <ChangeContent change={data.change} adopted={accepted.has(data.change.change_id)} terminal={terminal} />}
                  {step.truncated && !step.detail_available && <p>历史摘要已截断</p>}
                </>}
                </>}
              </div>}
            </div>
          </li>;
        })}</ol>
        {!steps.length && <p role="status">{terminal ? "没有执行步骤记录" : "正在准备核查…"}</p>}
      </div>
    </div>}
  </section>;
}

export function hasAgentResult(result) {
  return Boolean(result?.agent_result && result.agent_result.status !== "disabled");
}

function RelationSnapshot({ title, relations }) {
  return <div className="agent-snapshot"><Typography.Title heading={6}>{title}</Typography.Title>
    {relations.length ? relations.map((item, index) => <div className="agent-change" key={`${item.relation_id}-${index}`}>
      <p>{item.head} → {item.relation} → {item.tail}</p>
      <p>证据：{item.evidence || "-"}</p>
      <p>小节：{(item.source_sections || []).join("、") || "未定位"}；页码：{(item.source_pages || []).join("、") || "未提供"}</p>
    </div>) : <p>无</p>}
  </div>;
}

export function AgentRelationDetails({ relation, agent, onClose }) {
  const relationId = relation?.relation_id;
  const changes = (agent?.changes || []).filter((change) => change.relation_ids?.includes(relationId));
  const ids = new Set([...(relation?.verification?.task_ids || []), ...(changes.map((c) => c.task_id)), ...(relation?.task_id ? [relation.task_id] : [])]);
  const tasks = (agent?.tasks || []).filter((task) => ids.has(task.id) || task.relation_ids?.includes(relationId));
  return <Drawer title="关系核查详情" visible={Boolean(relation)} onCancel={onClose} footer={null} width="min(680px, 100vw)">
    {relation && <>
      <Tag>{relation.reason ? "已删除" : VERIFICATION_LABELS[relation.verification?.status] || "未核查"}</Tag>
      <Typography.Title heading={6}>{relation.head} → {relation.relation} → {relation.tail}</Typography.Title>
      {relation.reason && <p>删除理由：{relation.reason}</p>}
      {changes.map((change, index) => <Card size="small" key={index} className="agent-change" title={`改动 ${index + 1}`}>
        <RelationSnapshot title="改动前" relations={change.before || []} />
        <RelationSnapshot title="改动后" relations={change.after || []} />
        <p>原文依据：{change.evidence || change.reason || "-"}</p>
        <p>依据页码：{(change.pages || []).join("、") || "未提供"}</p>
      </Card>)}
      {tasks.length ? tasks.map((task) => <Card key={task.id} size="small" className="agent-change" title={`${TASK_LABELS[task.type] || "核查"} · ${TASK_STATUS_LABELS[task.status] || task.status}`}>
        <p>任务：{task.reason}</p><p>结论：{task.conclusion || "尚无结论"}</p>
        {(agent.trace || []).filter((event) => event.task_id === task.id && event.kind === "tool").map((event) => <div className="agent-tool" key={event.seq}>
          <strong>{TOOL_LABELS[event.tool] || event.tool}</strong>
          <p>输入：{Object.values(event.args || {}).map((value) => typeof value === "string" ? value : JSON.stringify(value)).join("；")}</p>
          <p>结果：{event.result}{event.truncated ? "（摘要已截断）" : ""}</p>
        </div>)}
      </Card>) : <Empty description="这条关系尚未被核查任务涉及" />}
    </>}
  </Drawer>;
}
