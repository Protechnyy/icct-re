# ICCT-RE 项目说明
## 1. 项目简介

ICCT-RE 是一个文档级关系抽取工作台：上传 PDF 或图片，PaddleOCR-VL 解析版面并重构文字，Skill4RE 根据文档内容选择领域 skill，调用兼容 OpenAI 接口的大模型抽取有原文证据的关系，最后展示并保存结果。当前内置军事、金融、法律、医疗、科技 5 个领域 skill。

处理链路：**文件上传 → Redis 任务队列 → Worker → OCR 与文档重构 → 分批关系抽取 → 结果合并与保存**。

## 2. 输入与输出

**输入**：通过网页上传，或向 `POST /api/upload` 发送 `multipart/form-data`。文件字段名为 `files`，可一次上传多个文件；后端支持 `.pdf`、`.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。网页文件选择器列出 PDF、PNG、JPG、JPEG。可选表单字段 `split_mode` 控制关系抽取粒度，默认 `small_section`（按小节）；还支持 `chapter`、`paragraph`、`fixed_sections`。`batch_size` 等细节参数见 `backend/.env.example`。

**上传响应**：HTTP 202，按文件返回任务 ID，例如：

```json
{
  "tasks": [
    {"task_id": "示例任务ID", "filename": "报告.pdf", "status": "queued", "progress": 0, "stage": "queued"}
  ]
}
```

**任务结果**：任务完成后，`GET /api/result/<task_id>` 返回包含 `document_meta`、`ocr_summary`、`document_text`、`final_relation_list`、`saved_paths` 等字段的 JSON。核心关系格式如下（示意数据）：

```json
{
  "final_relation_list": {
    "relation_list": [
      {
        "head": "第一机步营",
        "relation": "集结于",
        "tail": "西侧集结区",
        "evidence": "第一机步营在西侧集结区集结",
        "skill": "military",
        "source_sections": ["1.1"],
        "source_pages": [],
        "source_blocks": [],
        "source_paragraphs": [],
        "source_batch_index": 0
      }
    ]
  }
}
```

其中 `head`、`relation`、`tail` 是关系三元组，`evidence` 是原文证据，`skill` 是使用的领域技能。来源章节、页码、块和段落由后处理关联；无法定位时可能为空。完整响应还包含 OCR 和分批处理信息。结果同时写入 `data/results/<task_id>/`：`result.json`（完整结果）、`document_text.md`（重构文本）、`ocr_paragraphs.json`（OCR 段落）。`GET /api/result/<task_id>/csv` 可下载关系 CSV。

## 3. 部署与启动

以下是仓库当前的开发环境启动方式。需要 Python、Node.js 18+、Redis、PaddleOCR-VL 服务，以及一个可用的 OpenAI 兼容关系抽取模型接口；OCR 服务按当前 README 示例需要 GPU。Redis 默认地址为 `localhost:6379`，OCR 默认地址为 `http://127.0.0.1:8118/v1`。若从零启动依赖服务，可参考当前仓库使用的容器命令：

```bash
docker run -d --name docre-redis --restart unless-stopped \
  -p 6379:6379 m.daocloud.io/docker.io/library/redis:7

docker run -d --name docre-paddleocr --restart unless-stopped \
  --gpus all --network host -e PADDLE_PDX_MODEL_SOURCE=BOS \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:latest \
  paddlex_genai_server --model_name PaddleOCR-VL-1.6-0.9B \
  --host 0.0.0.0 --port 8118 --backend vllm
```

在仓库根目录准备后端环境：

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cd ..
```

编辑 `backend/.env`，至少核对 `REDIS_URL`、`PADDLE_OCR_SERVER_URL`、`VLLM_BASE_URL`、`VLLM_API_KEY`、`VLLM_MODEL`、`SKILL4RE_MODEL`。`VLLM_BASE_URL` 可填兼容接口的 `/v1` 地址或完整 `/v1/chat/completions` 地址；若模型名称变化，两个模型字段一起改。配置远程模型 API 即可，无需启动本地 vLLM；也可按 [README](../README.md) 切换为本地模型。

依赖服务就绪后，从仓库根目录分别启动：

```bash
./scripts/start-backend.sh
```

```bash
./scripts/start-frontend.sh
```

前一个脚本同时启动 Flask API 与 Worker；后一个脚本首次运行会安装前端依赖。访问 `http://127.0.0.1:5173`，健康检查为 `curl http://127.0.0.1:5000/api/health`。健康检查返回 `ok` 表示 Redis、OCR 和模型接口都可访问；`degraded` 表示至少一项不可用。

## 4. 任务如何运行

最简单的方式是在网页上传文件，等待任务状态变为 `succeeded`，然后查看关系结果或下载 CSV。命令行也可直接调用 API：

```bash
curl -X POST http://127.0.0.1:5000/api/upload \
  -F 'files=@/path/to/报告.pdf' \
  -F 'split_mode=small_section'
```

从响应中复制 `task_id`，再查询：

```bash
curl http://127.0.0.1:5000/api/status/<task_id>
curl http://127.0.0.1:5000/api/result/<task_id>
```

状态可能依次为 `queued`、`ocr_running`、`extracting`、`merging`、`succeeded`；若为 `failed`，查看状态响应的 `error` 字段。结果尚未生成时，结果接口返回 HTTP 409。

## 5. Prompt 编写指南

日常调整某个领域的抽取行为，编辑 `skill4re/skill4re/skills/<name>.json`（例如 `military.json`），或通过网页/`/api/skills` 接口维护。重点字段：

| 字段 | 写法 |
| --- | --- |
| `description`、`keywords` | 说明领域和触发词，用于技能路由。 |
| `focus`、`head_prior`、`tail_prior` | 写清要关注的关系，以及主体和客体的类型。 |
| `relation_style`、`negative_scope` | 约束关系词写法和不抽取的内容。 |
| `extraction_rules` | 用短句写具体抽取规则，明确关系方向、拆分条件、证据要求。 |
| `fewshot` | 给出原文片段及对应的 `relation_list` JSON，示范期望输出。 |

例如，新增 skill 的 JSON 可从下面模板开始；`fewshot.json` 写成 JSON 对象即可：

```json
{
  "name": "equipment",
  "description": "装备说明文档关系抽取",
  "focus": "装备配置、用途和部署位置",
  "head_prior": "装备或部件名称",
  "tail_prior": "用途、位置或配套部件",
  "relation_style": "使用短而明确的关系词，如装备、部署于、用于",
  "negative_scope": "不抽取没有原文证据的推测",
  "extraction_rules": ["每条关系保留原文证据", "装备与多个部件分别建立关系"],
  "keywords": ["装备", "部件", "部署"],
  "fewshot": [
    {
      "text": "甲型雷达部署于北区，用于目标探测。",
      "json": {
        "relation_list": [
          {"head": "甲型雷达", "relation": "部署于", "tail": "北区", "evidence": "甲型雷达部署于北区", "skill": "equipment"},
          {"head": "甲型雷达", "relation": "用于", "tail": "目标探测", "evidence": "用于目标探测", "skill": "equipment"}
        ]
      }
    }
  ]
}
```

公共提示词模板在 `skill4re/skill4re/prompts.py`：`BASE_EXTRACTION_RULES` 是通用规则，`build_router_prompt` 控制路由提示词，`build_extraction_prompt` 组合通用规则、选中 skill 的字段和 few-shot 示例；校对与合并提示词也在该文件中。当前路由和抽取模板的开头仍写有“军事”角色描述，新增非军事领域时应同步检查这些公共模板。修改公共模板后需重启后端 Worker；skill JSON 文件变化会在后续抽取任务中重新加载。编写时尽量保证：**原文有证据、关系方向明确、关系词具体、输出符合 `relation_list` JSON 格式**。

## 6. 公共提示词

整理自当前的 `skill4re/skill4re/prompts.py`。

引号和 JSON 大括号按模型实际看到的文本展示，不保留 Python 字符串的转义写法。`〈……〉` 表示运行时填入的文档、技能信息或关系结果；同一任务选中的 skill 不同，填入内容也不同。

### 6.1 公共抽取规则

`BASE_EXTRACTION_RULES` 会按下面的顺序进入抽取 prompt，后面再接所选 skill 的 `extraction_rules`：

```text
1. 只抽取原文有明确证据支持的关系，优先保证召回，后处理会去重。
2. 输出固定 JSON：{"relation_list":[{"head":"...","relation":"...","tail":"...","evidence":"...","skill":"..."}]}。
3. 字段必须使用中文或原文中文短语；relation 要短、稳定、语义明确，通常 2 到 8 个字。
4. head/tail 尽量是实体、组织、人物、设施、事件、概念或条件项，不要整句照抄原文。
5. 一句话含多个并列对象且证据充分时，拆成多条实体关系。
6. 不要输出摘要式泛关系，如 `X-实施-Y` `X-涉及-Y` 这类无信息量的写法。
7. evidence 必须是原文中可定位的片段，不要改写或概括。
8. 关系方向要自然：主动方做 head，被动方或目标做 tail。
9. 同一实体的不同表述统一为最完整的名称（如 `该营` 统一为 `第3机械化步兵营`）。
10. 只返回 JSON，不要 markdown，不要解释。
11. 【重要】关系谓词要精准，避免使用过泛的词：不要用 '必须'、'关键'、'威胁'、'涉及'、'包含' 等，改用更具体的动词如 '部署于'、'装备'、'依赖'、'导致'、'保护' 等。
12. 【重要】因果关系方向要正确：原因/条件做 head，结果/影响做 tail。例如：'基地车损毁-导致-行动中止' 而不是 '基地车-损毁于-行动失败'。
13. 【重要】基础元信息必须抽取：文件等级、作战区域、行动代号、行动时间、参战单位等。
```

### 6.2 技能路由 `build_router_prompt`

```text
你是一个军事文档技能路由器。请判断下面文档最适合哪些抽取 skills。

可选 skills：
〈每个 skill 的名称、描述、关注重点、关键词加权分数〉

要求：
1. 必须严格使用中文分析。
2. 只输出 JSON，不要解释，不要 markdown。
3. 输出格式：
{
  "primary_skill": "skill-name",
  "aux_skills": ["skill-name1", "skill-name2"],
  "reason": "中文简短说明"
}
4. `aux_skills` 最多两个，可以为空，总技能数最多 3 个。
5. 优先根据文档主要信息密度决定，而不是机械按关键词分数排序。
6. 如果文档同时明显包含兵力/指挥、阶段/约束、侦察火力/敌情作用三类信息，应覆盖这些主要语义面，不要只选一个抽象技能。

文档：
〈待路由的文档内容〉
```

### 6.3 关系抽取 `build_extraction_prompt`

```text
你是一个军事领域关系抽取系统。

你必须严格使用中文进行抽取，关系名、实体名、证据、skill 字段都必须是中文或原文中的中文短语，不要输出英文解释。

已匹配到以下 skills：
〈每个所选 skill 的名称、focus、head_prior、tail_prior、relation_style、negative_scope〉

few-shot 示例：
〈从所选 skill 中挑选的原文片段和 relation_list 输出示例，最多两个 few-shot 示例；没有时为“无”〉

输出要求：
`skill` 必须从以下值中选择：〈所选 skill 名称〉。
〈6.1 的公共抽取规则，再接所选 skill 的 extraction_rules，连续编号〉

文档：
〈待抽取的文档内容〉
```

这里的 `extraction_rules` 优先从 skill JSON 读取；只有该 skill 没有规则时，才使用 `prompts.py` 内的 `SKILL_EXTRACTION_RULES` 兜底。few-shot 示例根据文档内容与示例的相关性选择。

### 6.4 结果校对 `build_proofreading_prompt`

```text
你是一个军事关系抽取结果校对器。下面给你完整文档和一批候选关系，请在一次校对中同时完成整理、纠偏和少量补漏。

已选 skills：
〈每个所选 skill 的名称、关注重点、关系词风格、不应吸收的内容〉

校对要求：
1. 默认保留原有候选关系。只有明显重复、字段缺失、证据不支持、关系词明显错误、完全摘要化且已被更具体关系覆盖时，才能删除。
2. 不要把大量候选压缩成少量"最安全"的关系。候选能在原文中直接定位证据时，应优先保留或改写。
3. 保留开放域关系词，不要强行改成固定标签；relation 要短、稳定、军事语义明确。
4. 如果 `head` 或 `tail` 过长且像整句任务描述，请压缩成更短的实体、设施、线路、任务项或条件项。
5. 如果一句候选关系其实包含多个并列对象，且文档证据充分，可以拆成多条。
6. 如果某条关系更适合另一个已选 skill，可以改 `skill`；`skill` 只能从以下值中选择：〈所选 skill 名称〉。
7. 可以补充少量原文中直接有证据、但候选遗漏的重要关系，重点检查：
   - 阶段开始/结束时间
   - 阶段任务
   - 失败条件
   - 对象受限关系
   - 干扰压制对象
   - 火力覆盖/打击对象
   - 指挥控制链
   - 桥梁/道路/补给线节点作用
8. 优先保留实体之间、实体与设施/线路之间、任务项与约束条件之间的直接关系。
9. 去掉摘要式关系，如"我方-实施-联合突击""本次行动-主要目标-压制某目标"这类可被更具体关系替代的写法。
10. 保留阶段时间、失败条件、风险限制、敌火力/干扰作用对象等结构化信息，不要在校对时误删。
11. 输出仍必须严格为：
{
  "relation_list": [
    {
      "head": "...",
      "relation": "...",
      "tail": "...",
      "evidence": "...",
      "skill": "..."
    }
  ]
}
12. 只输出 JSON，不要解释，不要 markdown。

文档：
〈完整文档〉

候选关系：
〈包含 relation_list 的 JSON〉
```

### 6.5 跨片段合并 `build_summarize_prompt`

```text
你是一个军事关系抽取结果合并器。下面给你同一文档不同片段的抽取结果，请做跨片段的语义合并。

合并要求：
1. 同一实体的不同表述统一为最完整的名称（如"第3营"统一为"第3机械化步兵营"）。
2. 重复关系只保留证据最充分、表述最完整的一条。
3. 同一实体在不同片段中的不同关系都要保留（如片段A说"第3营部署于A高地"，片段B说"第3营负责主攻"，应保留两条）。
4. 不要新增文档中没有证据的关系。
5. 如果某条关系的 evidence 在多个片段中都出现，保留最短的那条。
6. skill 只能从以下值中选择：〈所选 skill 名称〉。
7. 输出格式：
{
  "relation_list": [
    {
      "head": "...",
      "relation": "...",
      "tail": "...",
      "evidence": "...",
      "skill": "..."
    }
  ]
}
8. 只输出 JSON，不要解释，不要 markdown。

文档：
〈完整文档〉

各片段抽取结果：
〈按 chunk_index 标记的 relation_list JSON 数组〉
```

### 6.6 低置信度复核 `build_targeted_proofreading_prompt`

```text
你是一个军事关系抽取结果反思器。以下是从文档中抽取的低置信度关系，请逐条审查并修正。

审查要求：
1. 对每条关系，检查 head、relation、tail 是否在文档中有明确证据支持。
2. 如果证据不足或关系明显错误，删除该条。
3. 如果实体名不准确（如用了简称而非全称），修正为文档中的完整表述。
4. 如果关系词不准确或过长，修正为更精准的短语。
5. 如果某条实际是正确的，保留它。
6. 不要新增文档中没有证据的关系。
7. skill 只能从以下值中选择：〈所选 skill 名称〉。
8. 输出格式：
{
  "relation_list": [
    {
      "head": "...",
      "relation": "...",
      "tail": "...",
      "evidence": "...",
      "skill": "..."
    }
  ]
}
9. 只输出 JSON，不要解释，不要 markdown。

文档：
〈完整文档〉

低置信度关系：
〈包含 relation_list 的 JSON〉
```
