import { Button, Empty, Segmented, Space, Tabs, Typography } from "antd";
import { DownloadOutlined } from "@ant-design/icons";
import { useMemo, useState } from "react";

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function highlightJson(jsonString) {
  return jsonString
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(
      /("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*")(\s*:)?/g,
      (match, stringPart, colon) => {
        if (colon) {
          return `<span class="json-key">${stringPart}</span><span class="json-colon">:</span>`;
        }
        return `<span class="json-string">${stringPart}</span>`;
      }
    )
    .replace(/\b(true|false)\b/g, '<span class="json-boolean">$1</span>')
    .replace(/\b(null)\b/g, '<span class="json-null">$1</span>')
    .replace(/\b(\d+\.?\d*)\b/g, '<span class="json-number">$1</span>');
}

function JsonBlock({ data }) {
  const pretty = useMemo(() => JSON.stringify(data, null, 2), [data]);
  const html = useMemo(() => highlightJson(pretty), [pretty]);
  return (
    <pre
      className="json-block"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export default function ResultViewer({ task, result }) {
  const [view, setView] = useState("final");

  if (!task) {
    return (
      <div className="panel panel-empty">
        <Empty description="选择一个任务查看抽取结果" />
      </div>
    );
  }

  const isReady = Boolean(result);
  const statusLabel = task.status === "succeeded" ? "已完成" : task.status === "failed" ? "失败" : "处理中…";
  const stageItems = result?.skill4re_result
    ? [
        {
          key: "routing",
          label: "路由",
          children: <JsonBlock data={result.stage_outputs?.routing || {}} />,
        },
        {
          key: "chunks",
          label: "分块",
          children: <JsonBlock data={result.stage_outputs?.chunk_predictions || []} />,
        },
        {
          key: "prediction",
          label: "抽取",
          children: <JsonBlock data={result.stage_outputs?.prediction || { relation_list: [] }} />,
        },
        {
          key: "timing",
          label: "耗时",
          children: <JsonBlock data={result.stage_outputs?.timing || {}} />,
        },
      ]
    : [
        {
          key: "sentence",
          label: "句级",
          children: <JsonBlock data={result?.stage_outputs?.sentence || []} />,
        },
        {
          key: "page",
          label: "页级",
          children: <JsonBlock data={result?.stage_outputs?.page || []} />,
        },
        {
          key: "multipage",
          label: "多页",
          children: <JsonBlock data={result?.stage_outputs?.multipage || []} />,
        },
      ];

  return (
    <div className="panel">
      <div className="panel-header result-header">
        <div style={{ minWidth: 0 }}>
          <Typography.Title level={4} style={{ margin: 0 }}>
            结果预览
          </Typography.Title>
          <Typography.Paragraph type="secondary" style={{ margin: "4px 0 0", fontSize: 13 }} ellipsis={{ tooltip: true }}>
            {task.filename} · {statusLabel}
          </Typography.Paragraph>
        </div>
        <Space wrap>
          <Segmented
            value={view}
            onChange={setView}
            options={[
              { label: "最终结果", value: "final" },
              { label: "阶段输出", value: "stage" },
              { label: "OCR 摘要", value: "ocr" },
            ]}
            size="small"
          />
          <Button
            type="primary"
            icon={<DownloadOutlined />}
            size="small"
            onClick={() => downloadJson(`${task.filename}.json`, result)}
            disabled={!isReady}
          >
            导出 JSON
          </Button>
        </Space>
      </div>

      {!isReady ? (
        <Empty description={task.status === "failed" ? "任务执行失败" : "结果尚未就绪，请稍候…"} />
      ) : view === "final" ? (
        <JsonBlock data={result.final_relation_list || { relation_list: result.final_relations || [] }} />
      ) : view === "stage" ? (
        <Tabs
          size="small"
          items={stageItems}
        />
      ) : (
        <JsonBlock
          data={{
            document_meta: result.document_meta,
            ocr_summary: result.ocr_summary,
            chunks: result.chunks,
          }}
        />
      )}
    </div>
  );
}
