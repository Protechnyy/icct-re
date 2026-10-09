import { Button, Card, Drawer, Empty, Space, Table, Tag, Typography } from "@arco-design/web-react";

export const AGENT_STATUS_LABELS = { completed: "完成", partial: "部分完成", failed: "失败", skipped: "跳过", disabled: "未开启" };
export const VERIFICATION_LABELS = { unchecked: "未核查", confirmed: "已确认", corrected: "已更正", added: "新增" };
const TASK_LABELS = { verify_evidence: "核查证据", normalize_entity: "归一实体", check_section: "检查小节漏抽" };
const TOOL_LABELS = { search_document: "检索原文", read_section: "读取小节", locate_evidence: "定位证据", query_entity: "查询实体", correct_relation: "更正关系", delete_relation: "删除关系", merge_entities: "合并实体", add_relation: "补充关系" };
const EVENT_LABELS = { plan: "规划", task_start: "任务开始", conclusion: "结论", task_end: "任务结束", error: "错误" };
const TASK_STATUS_LABELS = { pending: "待执行", running: "进行中", completed: "已完成", incomplete: "未完成", failed: "失败", not_executed: "未执行" };

export function hasAgentResult(result) {
  return Boolean(result?.agent_result && result.agent_result.status !== "disabled");
}

export function AgentProgress({ progress }) {
  if (!progress) return null;
  return <Card className="agent-progress" title="文档核查进度" size="small">
    <Typography.Text>已完成 {progress.completed_tasks || 0}/{progress.total_tasks || 0}</Typography.Text>
    {progress.current_task && <p>当前任务：{TASK_LABELS[progress.current_task.type] || "核查"} · {progress.current_task.reason || progress.current_task.target}</p>}
    <ol className="agent-events">{(progress.recent_events || []).map((event) => <li key={event.seq}>
      <Tag>{TOOL_LABELS[event.tool] || EVENT_LABELS[event.kind] || event.kind}</Tag>
      <span>{event.result}{event.truncated ? "（摘要已截断）" : ""}</span>
    </li>)}</ol>
  </Card>;
}

export function AgentSummary({ agent, onSelectRelation }) {
  const summary = agent.summary || {};
  const metrics = [["任务总数", summary.total_tasks], ["已确认", summary.confirmed_relations],
    ["已更正", summary.corrected_relations], ["新增", summary.added_relations],
    ["删除", summary.deleted_relations], ["实体合并", summary.entity_merges],
    ["模型调用", summary.llm_calls], ["令牌数", summary.tokens?.total_tokens]];
  return <Card className="agent-summary" title="文档核查摘要" size="small">
    <Space wrap><Tag color={agent.status === "completed" ? "green" : "orange"}>{AGENT_STATUS_LABELS[agent.status] || agent.status}</Tag>
      {metrics.map(([label, value]) => <span key={label}>{label}：{value ?? 0}</span>)}
      <span>耗时：{Number(summary.elapsed_seconds || 0).toFixed(2)} 秒</span>
    </Space>
    {agent.reason && <p>{agent.reason}</p>}
    {agent.planning?.degraded && <p>模型规划已降级为规则疑点任务。</p>}
    {agent.removed_relations?.length > 0 && <>
      <Typography.Title heading={6}>被删除的关系</Typography.Title>
      <Table size="small" rowKey="relation_id" pagination={false} data={agent.removed_relations} columns={[
        { title: "主体", dataIndex: "head" }, { title: "关系", dataIndex: "relation" },
        { title: "客体", dataIndex: "tail" }, { title: "删除理由", dataIndex: "reason" },
        { title: "核查过程", render: (_, item) => <Button type="text" size="mini" onClick={() => onSelectRelation(item)}>查看核查</Button> },
      ]} />
    </>}
  </Card>;
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
  return <Drawer title="关系核查详情" visible={Boolean(relation)} onCancel={onClose} footer={null} width={680}>
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
