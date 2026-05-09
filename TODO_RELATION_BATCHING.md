# 关系抽取分批优化 TODO

目标：把关系抽取从当前“固定每 2 个 Markdown 小节一批”改成可配置策略。默认策略应改为“一个小节一个 batch”，同时保留大章节、段落、固定 N 小节等模式。

## 1. 抽取粒度配置化

新增配置项：

```env
RELATION_SPLIT_MODE=small_section
RELATION_BATCH_SIZE=1
RELATION_MAX_BATCH_TOKENS=2500
RELATION_INCLUDE_PARENT_TITLE=true
```

支持模式：

- `small_section`：按 `1.1`、`1.2`、`4.4` 这种小节切，默认。
- `chapter`：按 `一、`、`二、`、`三、` 这种大章切。
- `paragraph`：按 Markdown 段落切，最细。
- `fixed_sections`：每 N 个小节一批，保留当前批处理能力。

## 2. 默认改成一个小节一抽

当前实际效果：

```text
4.3 + 4.4 一批
4.5 + 五 一批
```

目标默认效果：

```text
4.3 单独抽
4.4 单独抽
4.5 单独抽
五 单独抽
```

这样 `source_sections=["4.4"]` 更干净，错误定位和 evidence 回查更稳定。

## 3. 支持大章节模式

`chapter` 模式按大标题聚合：

```text
一、战区背景与态势
  1.1
  1.2
  1.3

二、任务目标
  2.1
  2.2
  2.3
```

注意：`Qwen3-8B` 的 `max_model_len=4096`，大章过长时必须自动降级拆小节。

## 4. 小节模式保留父标题上下文

小节单抽不能丢父标题。例如抽 `4.4` 时，输入应包含：

```text
四、行动阶段

4.4 第四阶段：实验室资料回收

正文...
```

## 5. 加 token 保护

每个 batch 生成后估算长度：

```env
RELATION_MAX_BATCH_TOKENS=2500
```

超过限制时自动降级：

```text
chapter -> small_section
small_section -> paragraph
paragraph -> hard split
```

避免再次触发 `http://192.168.1.2:9000/v1` 的 400。

## 6. API 上传时支持选择

前端上传时可带参数：

```text
split_mode=small_section
batch_size=1
```

后端 `/api/upload` 把参数写入 task payload，worker 按任务自己的配置执行。这样每次上传都能选，不需要改环境变量。

## 7. 前端页面加控件

上传区域添加：

```text
抽取粒度：
- 小节 1.1 / 4.4，默认
- 大章 一、二、三
- 段落
- 固定 N 小节

每批数量：
- 1，默认
- 2
- 3
```

如果选择 `chapter`，每批数量可以隐藏或置灰。

## 8. 结果里保存切分信息

`result.json` 保留：

```json
{
  "relation_split_config": {},
  "relation_sections": [],
  "relation_batches": []
}
```

每个 batch 记录：

```json
{
  "batch_index": 0,
  "split_mode": "small_section",
  "section_ids": ["4.4"],
  "parent_title": "四、行动阶段",
  "text": "..."
}
```

## 9. 关系结果带来源

每条关系继续保留：

```json
{
  "source_sections": ["4.4"],
  "source_batch_index": 0
}
```

后续如需精确页码和 bbox，再用 `evidence` 回查 `ocr_paragraphs.json`。

## 10. 测试

补测试：

- `small_section` 默认 1 个小节一批。
- `chapter` 模式能聚合 `1.1`、`1.2`、`1.3`。
- `batch_size=2` 时仍能两个小节一批。
- 超过 `RELATION_MAX_BATCH_TOKENS` 时会自动拆分。

## 建议默认值

```env
RELATION_SPLIT_MODE=small_section
RELATION_BATCH_SIZE=1
RELATION_MAX_BATCH_TOKENS=2500
RELATION_INCLUDE_PARENT_TITLE=true
```
