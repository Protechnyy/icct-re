import { Alert, Button, Card, Empty, Input, Message, Space, Table, Tabs, Tag, Tooltip, Typography } from "@arco-design/web-react";
import { IconCode, IconCopy, IconDownload, IconFile, IconRefresh, IconSearch } from "@arco-design/web-react/icon";
import { useMemo, useState } from "react";
import { exportTaskCsv } from "../lib/api";
import { STATUS_CONFIG } from "./TaskTable";

function downloadFile(filename, data, type = "application/json") {
  const blob = new Blob([typeof data === "string" ? data : JSON.stringify(data, null, 2)], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
    Message.success("已复制到剪贴板");
  } catch {
    Message.error("复制失败，请检查浏览器剪贴板权限");
  }
}

function JsonViewer({ data, filename }) {
  const pretty = useMemo(() => JSON.stringify(data, null, 2), [data]);
  return <div className="json-viewer">
    <div className="json-toolbar"><span><IconCode /> JSON 结果</span><Space size="mini"><Button size="mini" type="text" icon={<IconCopy />} onClick={() => copyText(pretty)}>复制</Button><Button size="mini" type="text" icon={<IconDownload />} onClick={() => downloadFile(filename, data)}>下载</Button></Space></div>
    <pre>{pretty}</pre>
  </div>;
}

function getRelations(result) {
  const source = result?.final_relation_list?.relation_list || result?.final_relation_list || result?.final_relations || result?.skill4re_result?.relation_list || [];
  return Array.isArray(source) ? source : [];
}

function uniqueValues(values) {
  return [...new Set((Array.isArray(values) ? values : [values]).filter((value) => value !== undefined && value !== null && value !== ""))];
}

function paragraphContents(paragraphs) {
  return uniqueValues((Array.isArray(paragraphs) ? paragraphs : []).map((paragraph) => paragraph?.content?.trim()));
}

function SourceParagraphs({ paragraphs }) {
  const contents = paragraphContents(paragraphs);
  if (!contents.length) return "-";
  const fullText = contents.join("\n\n");
  const summary = fullText.length > 64 ? `${fullText.slice(0, 64)}…` : fullText;
  return <Tooltip content={<div className="source-paragraph-tooltip">{fullText}</div>}><span className="source-paragraph-summary">{summary}</span></Tooltip>;
}

function getEntities(relations) {
  const entities = new Map();
  relations.forEach((item, index) => {
    const subject = item.subject || item.head || item.head_entity;
    const object = item.object || item.tail || item.tail_entity;
    [[subject, item.subject_type || item.head_type], [object, item.object_type || item.tail_type]].forEach(([name, type]) => {
      if (!name) return;
      const previous = entities.get(name);
      const sourceParagraphs = [...(previous?.sourceParagraphs || []), ...(Array.isArray(item.source_paragraphs) ? item.source_paragraphs : [])];
      entities.set(name, {
        key: previous?.key || `${name}-${index}`,
        name,
        type: previous?.type || type || "实体",
        sourceParagraphs,
      });
    });
  });
  return [...entities.values()];
}

function DocumentPreview({ result }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [zoom, setZoom] = useState(100);
  const text = result?.ocr_summary?.markdown_text || result?.document_meta?.markdown_text || result?.chunks?.map((chunk) => chunk.text).join("\n\n") || "当前任务未返回可预览的 OCR 原文。";
  const highlightedText = useMemo(() => {
    if (!query.trim()) return text;
    const parts = text.split(new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi"));
    return parts.map((part, index) => part.toLocaleLowerCase() === query.trim().toLocaleLowerCase() ? <mark key={`${part}-${index}`}>{part}</mark> : part);
  }, [query, text]);
  return <div className="document-preview"><div className="document-preview-toolbar"><Space size="small"><Button size="small" icon={<IconSearch />} onClick={() => setSearchOpen((current) => !current)}>搜索原文</Button><Button size="small" onClick={() => setZoom((current) => current >= 120 ? 90 : current + 10)}>{zoom}%</Button></Space><span>OCR 文本预览</span></div>{searchOpen && <div className="preview-search"><Input value={query} onChange={setQuery} allowClear placeholder="输入关键词，高亮原文匹配内容" /></div>}<article style={{ fontSize: `${zoom / 100}rem` }}>{highlightedText}</article></div>;
}

const STAGE_LABELS = {
  queued: "等待任务调度",
  layout_parsing: "版面解析",
  restructure_pages: "文档重构",
  relation_extraction: "关系抽取",
  document_merge: "结果合并",
  completed: "处理完成",
  failed: "处理失败",
};

const TIMING_LABELS = {
  total_elapsed_seconds: "总耗时",
  layout_parsing_seconds: "版面解析",
  document_restructure_seconds: "文档重构",
  relation_extraction_seconds: "关系抽取",
  document_merge_seconds: "结果合并",
  routing_seconds: "Skill 路由",
  coref_seconds: "指代消解",
  chunk_routing_seconds: "分块路由",
  extraction_seconds: "关系识别",
  summarize_seconds: "结果汇总",
  proofreading_seconds: "结果校验",
  refinement_seconds: "结果优化",
  domain_reflection_seconds: "领域复核",
  total_seconds: "抽取总耗时",
};

function formatDateTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value || "-" : date.toLocaleString("zh-CN", { hour12: false });
}

function formatDuration(value) {
  const seconds = Number(value);
  return Number.isFinite(seconds) ? `${seconds.toFixed(seconds < 10 ? 3 : 2)} 秒` : "-";
}

function ExecutionLog({ task, result }) {
  const timing = result?.stage_outputs?.timing || {};
  return <div className="execution-log">
    {task.error && <Alert type="error" title="任务错误" content={task.error} closable={false} />}
    <div className="log-row"><span>任务创建时间</span><span>{formatDateTime(task.created_at)}</span><Tag color="green">已记录</Tag></div>
    <div className="log-row"><span>当前阶段</span><span>{STAGE_LABELS[task.stage] || "处理中"}</span><Tag color={task.status === "failed" ? "red" : "arcoblue"}>{statusLabel(task.status)}</Tag></div>
    {Object.entries(timing).map(([stage, value]) => <div className="log-row" key={stage}><span>{TIMING_LABELS[stage] || "处理耗时"}</span><span>{formatDuration(value)}</span><Tag>耗时</Tag></div>)}
  </div>;
}

function statusLabel(status) {
  return STATUS_CONFIG[status]?.label || (status === "succeeded" ? "已完成" : "处理中");
}

function splitModeLabel(splitMode) {
  return {
    small_section: "小节",
    chapter: "章节",
    paragraph: "段落",
    fixed_sections: "固定长度",
  }[splitMode] || "小节";
}

export default function ResultViewer({ task, result }) {
  const [activeTab, setActiveTab] = useState("preview");
  const [exportingCsv, setExportingCsv] = useState(false);
  if (!task) return <Card className="result-empty-card"><Empty description={<div><div className="empty-title">选择一个任务查看抽取结果</div><div className="empty-description">从左侧任务列表中选择一个任务，查看文档内容和关系抽取结果。</div></div>} /></Card>;
  const status = STATUS_CONFIG[task.status] || { color: "gray", label: task.status || "未知" };
  const relations = getRelations(result);
  const entities = getEntities(relations);
  const entityColumns = [
    { title: "实体名称", dataIndex: "name", ellipsis: true },
    { title: "实体类型", dataIndex: "type", width: 120, render: (value) => <Tag color="arcoblue">{value}</Tag> },
    { title: "来源段落", width: 360, render: (_, item) => <SourceParagraphs paragraphs={item.sourceParagraphs} /> },
  ];
  const relationColumns = [
    { title: "主体", render: (_, item) => item.subject || item.head || item.head_entity || "-", ellipsis: true },
    { title: "关系", render: (_, item) => item.relation || "-", width: 150, ellipsis: true },
    { title: "客体", render: (_, item) => item.object || item.tail || item.tail_entity || "-", ellipsis: true },
    { title: "来源段落", render: (_, item) => <SourceParagraphs paragraphs={item.source_paragraphs} />, width: 360 },
  ];
  const tabItems = [
    { key: "preview", title: "文档预览", content: <DocumentPreview result={result} /> },
    { key: "entities", title: `实体 ${entities.length ? `(${entities.length})` : ""}`, content: <Table rowKey="key" columns={entityColumns} data={entities} pagination={false} scroll={{ x: 620 }} noDataElement={<Empty description="暂无实体结果" />} /> },
    { key: "relations", title: `关系 ${relations.length ? `(${relations.length})` : ""}`, content: <Table rowKey={(item, index) => item.id || `${index}-${item.relation}`} columns={relationColumns} data={relations} pagination={false} scroll={{ x: 720 }} noDataElement={<Empty description="暂无关系结果" />} /> },
    { key: "json", title: "JSON", content: <JsonViewer data={result || { status: task.status }} filename={`${task.filename}.json`} /> },
    { key: "logs", title: "运行日志", content: <ExecutionLog task={task} result={result} /> },
  ];
  const activeContent = tabItems.find((item) => item.key === activeTab)?.content || tabItems[0].content;
  async function handleExportCsv() {
    try {
      setExportingCsv(true);
      const blob = await exportTaskCsv(task.task_id);
      const stem = task.filename.replace(/\.[^.]+$/, "");
      downloadBlob(`${stem}_relations.csv`, blob);
      Message.success("CSV 导出成功");
    } catch (error) {
      Message.error(`CSV 导出失败：${String(error.message || error)}`);
    } finally {
      setExportingCsv(false);
    }
  }
  return <Card className="result-workspace" bordered={false}>
    <div className="task-summary-header">
      <div className="task-summary-main"><div className="summary-file-icon"><IconFile /></div><div><Tooltip content={task.filename}><Typography.Title heading={5} ellipsis={{ showTooltip: true }}>{task.filename}</Typography.Title></Tooltip><div className="task-summary-meta"><Tag color={status.color}>{status.label}</Tag><span>粒度：{splitModeLabel(task.metadata?.split_mode)}</span><span>Skill：自动路由</span><span>创建于 {task.created_at ? new Date(task.created_at).toLocaleString("zh-CN") : "-"}</span></div></div></div>
      <Space wrap><Tooltip content="当前后端暂未提供重新执行接口"><Button icon={<IconRefresh />} disabled>重新执行</Button></Tooltip><Button icon={<IconDownload />} onClick={() => result && downloadFile(`${task.filename}.json`, result)} disabled={!result}>导出 JSON</Button><Button type="primary" icon={<IconDownload />} loading={exportingCsv} disabled={!result} onClick={handleExportCsv}>导出 CSV</Button></Space>
    </div>
    {!result && task.status === "failed" ? <Alert type="error" title="任务执行失败" content={task.error || "请检查运行日志后重新执行任务。"} closable={false} /> : null}
    {!result && task.status !== "failed" ? <div className="result-pending"><Empty description="结果尚未就绪，系统将持续更新任务进度。" /></div> : <>
      <Tabs activeTab={activeTab} onChange={setActiveTab} type="line" animation={false}>
        {tabItems.map((item) => <Tabs.TabPane key={item.key} title={item.title} />)}
      </Tabs>
      <div className={`result-tab-body ${activeTab === "json" ? "is-json" : ""}`}>
        {activeContent}
      </div>
    </>}
  </Card>;
}
