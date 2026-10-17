import Button from "@arco-design/web-react/es/Button";
import Card from "@arco-design/web-react/es/Card";
import InputNumber from "@arco-design/web-react/es/InputNumber";
import Radio from "@arco-design/web-react/es/Radio";
import Slider from "@arco-design/web-react/es/Slider";
import Upload from "@arco-design/web-react/es/Upload";
import { IconDelete, IconFile, IconFileImage, IconFilePdf, IconUpload } from "@arco-design/web-react/icon";
import "@arco-design/web-react/es/InputNumber/style/css.js";
import "@arco-design/web-react/es/Radio/style/css.js";
import "@arco-design/web-react/es/Slider/style/css.js";
import "@arco-design/web-react/es/Upload/style/css.js";

// 思考强度滑杆档位：档位值对应 thinking_budget，0 档 budget=0 表示不限制模型思考长度
const THINKING_BUDGET_OPTIONS = [
  { label: "低", budget: 1024 },
  { label: "中", budget: 4096 },
  { label: "高", budget: 0 },
];

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
  const streamThinking = relationOptions?.stream_thinking !== false;
  const thinkingBudget = relationOptions?.thinking_budget ?? 1024;
  const budgetIndex = Math.max(0, THINKING_BUDGET_OPTIONS.findIndex((option) => option.budget === thinkingBudget));
  const budgetLabel = THINKING_BUDGET_OPTIONS[budgetIndex].label;
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
        <button type="button" className="upload-trigger" aria-label="选择文档文件">
          <IconUpload className="upload-icon" />
          <div className="upload-title">点击或拖拽文件到此处上传</div>
          <div className="upload-hint">支持 PDF、PNG、JPG、JPEG，可批量上传</div>
        </button>
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
              <Button type="text" status="danger" size="mini" icon={<IconDelete />} aria-label={`移除文件：${file.name}`} onClick={() => onRemove(file)} />
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
          <Radio value="fixed_sections">固定小节数</Radio>
        </Radio.Group>
        {splitMode === "fixed_sections" && (
          <div className="fixed-length-row">
            <label htmlFor="fixed-section-count">每批小节数</label>
            <InputNumber id="fixed-section-count" min={1} max={Number.MAX_SAFE_INTEGER} precision={0} value={batchSize} onChange={(value) => updateOptions({ batch_size: value || 1 })} suffix="小节" />
          </div>
        )}
      </div>
      <div className="task-config thinking-config-row">
        <div className="thinking-field">
          <div className="field-label thinking-label-row"><span>思考强度</span><span className="thinking-budget-value">{budgetLabel}</span></div>
          <Slider
            className="thinking-budget-slider"
            min={0}
            max={THINKING_BUDGET_OPTIONS.length - 1}
            step={1}
            value={budgetIndex}
            formatTooltip={(value) => THINKING_BUDGET_OPTIONS[value]?.label ?? value}
            onChange={(value) => updateOptions({ thinking_budget: THINKING_BUDGET_OPTIONS[value].budget })}
            aria-label="思考强度"
          />
        </div>
        <div className="thinking-field">
          <div className="field-label">思考过程</div>
          <Radio.Group
            type="button"
            value={streamThinking ? "on" : "off"}
            onChange={(value) => updateOptions({ stream_thinking: value === "on" })}
            aria-label="输出思考过程"
          >
            <Radio value="on">开启</Radio>
            <Radio value="off">关闭</Radio>
          </Radio.Group>
        </div>
      </div>
      <div className="extraction-actions">
        <Button className="submit-task-button" type="primary" icon={<IconUpload />} loading={submitting} disabled={!fileList.length} onClick={onSubmit}>开始抽取</Button>
      </div>
    </Card>
  );
}
