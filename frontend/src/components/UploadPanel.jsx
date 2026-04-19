import { DeleteOutlined, FileImageOutlined, FilePdfOutlined, FileTextOutlined, InboxOutlined } from "@ant-design/icons";
import { Button, Space, Tag, Typography } from "antd";
import { useCallback, useRef, useState } from "react";

function formatSize(bytes) {
  if (!bytes && bytes !== 0) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function fileIcon(file) {
  const type = file.type || "";
  if (type.includes("pdf")) return <FilePdfOutlined />;
  if (type.startsWith("image/")) return <FileImageOutlined />;
  return <FileTextOutlined />;
}

export default function UploadPanel({ fileList, onChange, onSubmit, onRemove, submitting }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const makeFileItem = (file) => ({
    uid: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
    name: file.name,
    size: file.size,
    type: file.type,
    originFileObj: file,
  });

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    const dropped = Array.from(e.dataTransfer.files || []).filter((f) =>
      /\.(pdf|png|jpg|jpeg|webp|bmp)$/i.test(f.name)
    );
    if (!dropped.length) return;
    const next = [...fileList, ...dropped.map(makeFileItem)];
    onChange?.({ fileList: next });
  }, [fileList, onChange]);

  const handleInputChange = useCallback((e) => {
    const selected = Array.from(e.target.files || []);
    if (!selected.length) return;
    const next = [...fileList, ...selected.map(makeFileItem)];
    onChange?.({ fileList: next });
    e.target.value = "";
  }, [fileList, onChange]);

  return (
    <div className="panel upload-panel">
      <div className="panel-header">
        <Typography.Title level={4}>上传文档</Typography.Title>
        <Typography.Paragraph type="secondary" style={{ margin: 0, fontSize: 13 }}>
          支持 PDF 和常见图片格式，批量上传后会自动创建独立任务。
        </Typography.Paragraph>
      </div>

      <div
        className={`upload-dropzone ${dragOver ? "active" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.png,.jpg,.jpeg,.webp,.bmp"
          style={{ display: "none" }}
          onChange={handleInputChange}
        />
        <p className="upload-dropzone-icon"><InboxOutlined /></p>
        <p className="upload-dropzone-text">点击或拖拽文件到此处</p>
        <p className="upload-dropzone-hint">
          支持 PDF、PNG、JPG、WEBP、BMP 格式
          <br />
          选中文件后会自动开始上传，后端依次执行 PaddleOCR-VL 与 Qwen 关系抽取
        </p>
      </div>

      <div className="upload-toolbar">
        <Space wrap>
          <Button
            type="primary"
            onClick={onSubmit}
            loading={submitting}
            disabled={!fileList.length}
          >
            重新上传当前选择
          </Button>
          <Typography.Text type="secondary" style={{ fontSize: 13 }}>
            已选择 {fileList.length} 个文件
          </Typography.Text>
          {!fileList.length ? (
            <Tag color="default" style={{ fontSize: 12 }}>选择文件后会自动开始上传</Tag>
          ) : null}
        </Space>
      </div>

      <div className="upload-selection">
        {fileList.length ? (
          fileList.map((item) => (
            <div className="file-list-card" key={item.uid}>
              <span className="file-icon">{fileIcon(item)}</span>
              <div className="file-info">
                <div className="file-name" title={item.name}>
                  {item.name}
                </div>
                <div className="file-meta">
                  {formatSize(item.size)} · {item.type || "未知格式"}
                </div>
              </div>
              <span
                className="file-remove"
                title="移除"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove?.(item);
                }}
              >
                <DeleteOutlined />
              </span>
            </div>
          ))
        ) : (
          <Typography.Text type="secondary" style={{ fontSize: 13 }}>
            当前还没有已选文件。
          </Typography.Text>
        )}
      </div>
    </div>
  );
}
