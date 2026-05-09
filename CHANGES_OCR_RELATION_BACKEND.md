# OCR、关系抽取与后端改动说明

本文记录当前项目相对原始 `icct-re` 的主要修改点，重点包括 OCR 接入方式、关系抽取分批方式、后端运行与结果保存逻辑。

## 1. OCR 识别改动

原项目 OCR 方案偏向 Docker / 本地 `genai_server` 服务。当前已改成后端直接使用 `PaddleOCRVL` Python API，并通过远程 OpenAI 兼容服务进行 VL 识别。

当前 OCR 配置：

```env
PADDLE_OCR_MODE=python_api
PADDLE_OCR_BASE_URL=http://192.168.1.2:8080
PADDLE_OCR_SERVER_URL=http://192.168.1.2:8080/v1
PADDLE_OCR_API_MODEL_NAME=PaddleOCR-VL-1.5-0.9B
PADDLE_OCR_API_KEY=EMPTY
PADDLE_OCR_MAX_CONCURRENCY=32
```

关键点：

- 本机不安装 `vllm`。
- 本机不安装 PaddlePaddle GPU。
- 后端环境使用 CPU Paddle：`paddlepaddle==3.3.1`。
- VL 识别请求走远程：`http://192.168.1.2:8080/v1`。
- OCR 结果优先读取 `parsing_res_list`，按 PaddleOCR-VL 已分好的块/段落组织文本。
- 过滤页码、页眉、页脚、旁注、印章等无效块。
- 输出保存为：
  - `data/results/<task_id>/document_text.md`
  - `data/results/<task_id>/ocr_paragraphs.json`
  - `data/results/<task_id>/result.json`

相关文件：

- `backend/app/paddle_ocr.py`
- `backend/app/config.py`
- `backend/.env.example`
- `backend/requirements.txt`
- `backend/tests/test_paddle_ocr.py`

## 2. 关系抽取方式改动

原流程是 OCR 后把整篇 `document_text` 一次性交给 Skill4RE 做关系抽取。

当前流程改成：

```text
OCR
-> document_text / Markdown 正文
-> 按 Markdown 标题切 section
-> 分 batch 调用 Skill4RE
-> 合并去重
-> 保存 final_relations
```

当前默认配置：

```env
VLLM_BASE_URL=http://192.168.1.2:9000/v1
VLLM_MODEL=Qwen3-8B
SKILL4RE_MODEL=Qwen3-8B
RELATION_SECTION_BATCH_SIZE=2
```

注意：

- 关系抽取 LLM 使用远程 `http://192.168.1.2:9000/v1`。
- 模型为 `Qwen3-8B`。
- 该模型 `max_model_len=4096`，所以之前 3 个 section 一批容易触发 400。
- 当前临时稳定方案是每 2 个 Markdown section 一批。
- 后续计划改为默认 1 个小节一批，详见 `TODO_RELATION_BATCHING.md`。

结果位置：

```text
data/results/<task_id>/result.json
```

关系抽取结果字段：

```json
final_relations
```

同一份包装版：

```json
final_relation_list.relation_list
```

每条关系包含：

```json
{
  "head": "...",
  "relation": "...",
  "tail": "...",
  "evidence": "...",
  "skill": "...",
  "source_sections": ["4.4"],
  "source_pages": [],
  "source_blocks": [],
  "source_batch_index": 0
}
```

相关文件：

- `backend/app/pipeline.py`
- `skill4re/skill4re/service.py`
- `TODO_RELATION_BATCHING.md`

## 3. 后端改动

是的，后端也修改了。主要改动如下。

### 3.1 配置默认值

`backend/app/config.py` 默认值已切换为：

```env
PADDLE_OCR_BASE_URL=http://192.168.1.2:8080
PADDLE_OCR_SERVER_URL=http://192.168.1.2:8080/v1
PADDLE_OCR_API_MODEL_NAME=PaddleOCR-VL-1.5-0.9B
VLLM_BASE_URL=http://192.168.1.2:9000/v1
VLLM_MODEL=Qwen3-8B
```

### 3.2 API 启动修正

`backend/run_api.py` 原来强制 `debug=True`，容易触发 Flask reloader，不适合后台启动。

现在改成读取配置：

```python
app.run(host=config.api_host, port=config.api_port, debug=config.debug, use_reloader=False)
```

### 3.3 结果落盘

`backend/app/pipeline.py` 增加了结果保存逻辑：

```text
data/results/<task_id>/result.json
data/results/<task_id>/document_text.md
data/results/<task_id>/ocr_paragraphs.json
```

### 3.4 Python 兼容修复

`backend/app/utils.py` 修复了 Python 3.10 下 `datetime.UTC` 不兼容问题，改为 `timezone.utc`。

### 3.5 依赖调整

`backend/requirements.txt` 当前策略：

- 保留 `paddleocr[doc-parser]>=3.4,<3.5`
- 使用 `paddlepaddle==3.3.1`
- 不包含 `vllm`
- 不包含 `paddlepaddle-gpu`
- 不包含 CUDA / NVIDIA 大包

## 4. 前端与一键启动

新增：

```text
start.sh
```

用途：

- 启动后端 API。
- 启动 worker。
- 启动前端 Vite 页面。
- 自动设置 OCR / LLM 环境变量。
- 如果前端缺少 `vite`，自动执行 `npm install`。

启动：

```bash
cd /mnt/code3/recctt/icct-re
./start.sh
```

停止：

```bash
cd /mnt/code3/recctt/icct-re
./start.sh stop
```

页面：

```text
http://127.0.0.1:5173
```

日志：

```text
data/tmp/api-5011.log
data/tmp/worker.log
data/tmp/frontend-5173.log
```

## 5. 已验证结果

测试 PDF：

```text
铁幕回声行动_完整作战文档.pdf
```

OCR 结果：

```text
page_count=4
raw_block_count=46
paragraph_count=46
document_chars=3347
```

正式队列跑通任务：

```text
task_id=d1d7385fb83c4c8dbb44cd4963ee40b8
status=succeeded
relation_count=120
relation_section_count=18
relation_batch_count=9
```

结果文件：

```text
data/results/d1d7385fb83c4c8dbb44cd4963ee40b8/result.json
```

## 6. 后续优化方向

下一步建议按 `TODO_RELATION_BATCHING.md` 做：

- 默认改为 1 个小节一抽。
- 支持小节、大章、段落、固定 N 小节等策略。
- 前端上传时可选择抽取粒度。
- 加 token 保护，避免 `Qwen3-8B` 超上下文。
- relation 继续保留 `source_sections`，后续可用 evidence 回查页码和 bbox。
