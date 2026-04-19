import { CheckCircleOutlined, CloseCircleOutlined, FileTextOutlined, LoadingOutlined, SyncOutlined } from "@ant-design/icons";
import { Progress, Table, Tag, Tooltip, Typography } from "antd";

const STATUS_CONFIG = {
  queued: { color: "default", icon: null, label: "排队中" },
  ocr_running: { color: "processing", icon: <SyncOutlined spin />, label: "OCR 识别" },
  extracting: { color: "processing", icon: <SyncOutlined spin />, label: "关系抽取" },
  merging: { color: "processing", icon: <SyncOutlined spin />, label: "结果整合" },
  succeeded: { color: "success", icon: <CheckCircleOutlined />, label: "已完成" },
  failed: { color: "error", icon: <CloseCircleOutlined />, label: "失败" },
};

export default function TaskTable({ tasks, onSelectTask, activeTaskId }) {
  const columns = [
    {
      title: "文件名",
      dataIndex: "filename",
      key: "filename",
      width: 160,
      ellipsis: true,
      render: (value, record) => (
        <Tooltip title={value}>
          <button
            className={`row-link ${activeTaskId === record.task_id ? "active" : ""}`}
            onClick={() => onSelectTask(record.task_id)}
          >
            <FileTextOutlined style={{ marginRight: 6, opacity: 0.65 }} />
            {value}
          </button>
        </Tooltip>
      ),
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      width: 90,
      render: (value) => {
        const cfg = STATUS_CONFIG[value] || { color: "default", icon: null, label: value };
        return (
          <Tag color={cfg.color} icon={cfg.icon} style={{ fontSize: 12 }}>
            {cfg.label}
          </Tag>
        );
      },
    },
    {
      title: "进度",
      dataIndex: "progress",
      key: "progress",
      render: (value, record) => (
        <Tooltip title={record.error || undefined}>
          <Progress percent={value || 0} size="small" />
        </Tooltip>
      ),
    },
  ];

  return (
    <div className="panel task-table">
      <div className="panel-header">
        <Typography.Title level={4} style={{ margin: 0 }}>
          任务列表
        </Typography.Title>
      </div>
      <Table
        rowKey="task_id"
        columns={columns}
        dataSource={tasks}
        pagination={false}
        size="small"
        scroll={{ x: false, y: '100%' }}
        locale={{ emptyText: "暂无任务" }}
      />
    </div>
  );
}
