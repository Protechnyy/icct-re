# OCR、关系抽取与后端改动说明

本文记录本次整合后的主要行为变化，重点包括 OCR 结果整理、关系抽取分批、结果落盘和启动脚本。

## OCR 识别

后端继续通过 `PaddleOCRVL` Python API 调用 OpenAI 兼容的 VL 识别服务。服务地址、模型名和并发度都通过环境变量配置：

```env
PADDLE_OCR_SERVER_URL=http://127.0.0.1:8118/v1
PADDLE_OCR_API_MODEL_NAME=PaddleOCR-VL-1.5-0.9B
PADDLE_OCR_MAX_CONCURRENCY=4
```

OCR 结果优先读取 `parsing_res_list`，按 PaddleOCR-VL 已分好的块组织正文，并过滤页码、页眉、页脚、旁注、印章等无效块。

## 关系抽取分批

OCR 后的 Markdown 正文会先切成 section，再按配置分 batch 调用 Skill4RE：

```env
RELATION_SPLIT_MODE=small_section
RELATION_BATCH_SIZE=1
RELATION_MAX_BATCH_TOKENS=2500
RELATION_INCLUDE_PARENT_TITLE=true
```

支持模式：

- `small_section`：按 `1.1`、`4.4` 等小节切分
- `chapter`：按 `一、`、`二、` 等大章切分
- `paragraph`：按 Markdown 段落切分
- `fixed_sections`：每 N 个小节一批

每条关系会带上来源信息，例如 `source_sections`、`source_batch_index`、`source_pages` 和 `source_blocks`。

## 结果落盘

任务完成后会保存：

```text
data/results/<task_id>/result.json
data/results/<task_id>/document_text.md
data/results/<task_id>/ocr_paragraphs.json
```

API 返回中也包含 `saved_paths`，方便从任务结果定位落盘文件。

## 启动脚本

新增 [start.sh](start.sh)，用于从仓库根目录同时启动 API、Worker 和前端。脚本默认使用 `backend/.venv/bin/python`，如果不存在则回退到 `python3`。

所有服务地址、端口、模型名和切分参数都可以通过环境变量覆盖，不需要修改源码。
