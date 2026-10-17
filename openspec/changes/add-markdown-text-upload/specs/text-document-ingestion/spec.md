# Spec Delta

## Purpose

接收已经包含文字内容的 Markdown 与纯文本文档，让系统跳过 OCR 并将原文送入现有关系抽取流程，同时保留 PDF 和图片的处理方式。

## ADDED Requirements

### Requirement: 支持 Markdown 和纯文本上传

系统 SHALL 在现有 PDF 和图片格式之外接受 Markdown（`.md`、`.markdown`）和纯文本（`.txt`）文件。扩展名匹配 MUST 忽略大小写，界面和服务端 MUST 使用一致的支持格式。

#### Scenario: 上传 Markdown 文档
- **WHEN** 用户选择或提交 `.md` 或 `.markdown` 文件
- **THEN** 系统接受文件并使用原文件名创建任务

#### Scenario: 上传纯文本文件
- **WHEN** 用户选择或提交 `.txt` 文件
- **THEN** 系统接受文件并使用原文件名创建任务

#### Scenario: 上传不支持的格式
- **WHEN** 用户提交未支持的文件格式
- **THEN** 系统拒绝文件并返回清楚的格式错误

### Requirement: 读取文本文件编码

系统 MUST 将 Markdown 和 TXT 按 UTF-8 解码，并移除可选的 UTF-8 BOM。空白文件或无效 UTF-8 文件 MUST 在模型抽取前以明确的文件错误终止任务。

#### Scenario: 读取带 BOM 的 UTF-8 文件
- **WHEN** 上传文件以 UTF-8 BOM 开头
- **THEN** 系统移除 BOM 并读取原文内容

#### Scenario: 读取空白或编码无效的文件
- **WHEN** 上传文件内容为空白，或无法按 UTF-8 解码
- **THEN** 任务失败并说明对应文件无法读取，且不调用模型抽取

### Requirement: 文本文件绕过 OCR 并进入抽取流程

系统 MUST 直接读取 Markdown 和 TXT 内容，MUST NOT 调用 PaddleOCR 版面解析或文档重构。读取成功的文本 MUST 继续进入现有 Skills 路由、关系抽取、可选的文档校验及结果保存流程。

#### Scenario: 处理 Markdown 或 TXT 任务
- **WHEN** Worker 处理有效的 Markdown 或 TXT 任务
- **THEN** 任务跳过 OCR 状态和 OCR 解析步骤，使用读取的文本生成关系抽取结果

#### Scenario: 处理 PDF 或图片任务
- **WHEN** Worker 处理现有支持的 PDF 或图片任务
- **THEN** 系统继续使用 PaddleOCR 版面解析和现有文档重构流程

### Requirement: 保留文本内容与 Markdown 结构

系统 MUST 在结果和原文预览中提供读取到的文档文本。Markdown 标题和段落边界 MUST 保留并参与现有章节切分；Markdown 中引用的图片 MUST NOT 被读取或提交给 OCR。

#### Scenario: 预览 Markdown 文档
- **WHEN** 用户查看 Markdown 任务结果
- **THEN** 原文预览保留其文本与标题结构，章节标题参与关系抽取分段

#### Scenario: Markdown 包含图片引用
- **WHEN** Markdown 原文包含本地或远程图片引用
- **THEN** 系统只处理文本内容，不读取图片文件或调用 PaddleOCR
