import { Button, Card, InputNumber, Radio, Upload } from "@arco-design/web-react";
import { IconDelete, IconFile, IconFileImage, IconFilePdf, IconUpload } from "@arco-design/web-react/icon";

function formatSize(bytes) {
  if (bytes === undefined || bytes === null) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function fileIcon(file) {
  const type = file.type || file.originFile?.type || "";
  if (type.includes("pdf") || /\.pdf$/i.test(file.name)) return <IconFilePdf />;
  if (type.startsWith("image/") || /\.(png|jpe?g)$/i.test(file.name)) return <IconFileImage />;
  return <IconFile />;
}

export default function UploadPanel({ fileList, onChange, onSubmit, onRemove, submitting, relationOptions, onRelationOptionsChange }) {
  const splitMode = relationOptions?.split_mode || "small_section";
  const batchSize = relationOptions?.batch_size || 1;
  const updateOptions = (next) => onRelationOptionsChange({ ...relationOptions, ...next });

  return (
    <Card className="task-creator-card" title="新建抽取任务" bordered>
      <div className="field-label">文件上传</div>
      <Upload
        drag
        autoUpload={false}
        multiple
        accept=".pdf,.png,.jpg,.jpeg"
        fileList={fileList}
        showUploadList={false}
        onChange={(nextFileList) => onChange(nextFileList)}
        className="document-uploader"
      >
        <div className="upload-trigger">
          <IconUpload className="upload-icon" />
          <div className="upload-title">点击或拖拽文件到此处上传</div>
          <div className="upload-hint">支持 PDF、PNG、JPG、JPEG，可批量上传</div>
        </div>
      </Upload>
      {fileList.length > 0 && (
        <div className="selected-file-list">
          <div className="file-count">已选择 {fileList.length} 个文件</div>
          {fileList.map((file) => (
            <div className="selected-file" key={file.uid}>
              <span className="selected-file-icon">{fileIcon(file)}</span>
              <div className="selected-file-detail">
                <span title={file.name} className="selected-file-name">{file.name}</span>
                <span className="selected-file-size">{formatSize(file.size)}</span>
              </div>
              <Button type="text" status="danger" size="mini" icon={<IconDelete />} onClick={() => onRemove(file)} />
            </div>
          ))}
        </div>
      )}
      <div className="task-config">
        <div className="field-label">抽取粒度</div>
        <Radio.Group type="button" value={splitMode} onChange={(value) => updateOptions({ split_mode: value, batch_size: value === "fixed_sections" ? batchSize : 1 })}>
          <Radio value="small_section">小节</Radio>
          <Radio value="chapter">章节</Radio>
          <Radio value="paragraph">段落</Radio>
          <Radio value="fixed_sections">固定长度</Radio>
        </Radio.Group>
        {splitMode === "fixed_sections" && (
          <div className="fixed-length-row">
            <span>每段最大长度</span>
            <InputNumber min={1} max={18000} value={batchSize} onChange={(value) => updateOptions({ batch_size: value || 1 })} suffix="字符" />
          </div>
        )}
      </div>
      <Button className="submit-task-button" type="primary" long icon={<IconUpload />} loading={submitting} disabled={!fileList.length} onClick={onSubmit}>开始抽取</Button>
    </Card>
  );
}
