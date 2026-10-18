# Proposal

## Why

当前上传入口只接受 PDF 和图片，所有任务都会进入 PaddleOCR 版面解析。Markdown 和纯文本已经包含可抽取的文字，调用 OCR 会增加处理时间和服务依赖，也可能改变原文格式。

## What Changes

- 上传界面和后端接受 Markdown（`.md`、`.markdown`）及纯文本（`.txt`）文件，并保留现有 PDF 和图片支持。
- 后端直接读取文本文件内容，跳过 PaddleOCR 版面解析和文档重构，继续使用现有关系抽取、Skills 路由和结果保存流程。
- 结果预览呈现输入文本；Markdown 标题继续参与现有章节切分。
- 文本文件按 UTF-8 读取并兼容 BOM；无法按该编码读取时向任务提供明确错误。

## Capabilities

### New Capabilities

- `text-document-ingestion`：接收 Markdown 和纯文本文件，直接读取内容并送入文档关系抽取流程。

### Modified Capabilities

无。

## Impact

- 后端：上传格式识别、文本读取、文档处理阶段及结果元数据。
- 前端：上传类型提示、文件选择限制、任务阶段和原文预览文案。
- 测试：文件类型识别、文本读取、OCR 跳过、现有抽取结果兼容。
