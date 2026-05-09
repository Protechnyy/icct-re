# Skill4RE

基于技能路由的军事文档关系抽取系统，从 OneKE 项目中独立迭代而来。

## 核心特性

- **技能路由**：关键词 + LLM 混合路由，自动选择最适合的抽取技能
- **多领域支持**：军事、金融、医疗、法律、科技 5 大领域
- **模块化设计**：技能定义与执行逻辑解耦，易于扩展
- **长文档处理**：支持文档分块、共指消解、跨块语义合并
- **置信度评估**：低置信度关系自动触发 LLM 反思修正

## 目录结构

```
skill4re/
├── skills/              # 领域技能定义（JSON）
│   ├── military.json    # 军事领域
│   ├── finance.json     # 金融领域
│   ├── medical.json     # 医疗领域
│   ├── legal.json       # 法律领域
│   └── technology.json  # 科技领域
├── prompts.py           # 路由/抽取/校对提示词
├── service.py           # 关系抽取服务核心
├── routing.py           # 关键词 + LLM 混合路由
├── backends.py          # API/本地模型后端
├── normalization/       # 结果归一化
│   ├── dedup.py         # 去重逻辑
│   ├── confidence.py    # 置信度评估
│   ├── entity.py        # 实体归一化
│   └── evidence.py      # 证据校验
├── coref.py             # 共指消解
├── dataset.py           # 数据加载与分块
├── models.py            # 数据模型定义
├── types.py             # 类型定义
├── config.py            # 配置常量
├── loader.py            # 技能加载
├── parsing.py           # JSON 解析
└── run.py               # CLI 入口
```

## 抽取流程

1. **文档级路由**：根据关键词分数 + LLM 判断选择技能组合
2. **文档分块**：长文档自动切块，每块独立路由
3. **关系抽取**：使用选中技能的专项规则 + few-shot 示例
4. **结果合并**：跨块语义合并（LLM 或规则）
5. **置信度反思**：低置信度关系触发 LLM 审查修正

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行抽取

```bash
# 使用 Qwen API
python skill4re/run.py --backend qwen_api --model qwen3-32b --limit 1

# 使用本地 vLLM OpenAI 兼容服务
python skill4re/run.py --backend vllm --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-32B-AWQ --limit 1

# 使用 OpenAI API
python skill4re/run.py --backend openai --model gpt-4o-mini --limit 1

# 指定输出路径
python skill4re/run.py \
  --backend qwen_api \
  --model qwen3-32b \
  --limit 1 \
  --output ./output/result.json
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--backend` | 后端类型：openai/api/qwen_api/local_qwen3/vllm | api |
| `--base-url` | vLLM OpenAI 兼容服务地址 | http://127.0.0.1:8000/v1 |
| `--model` | 模型名称 | deepseek-chat |
| `--limit` | 处理样本数 | 5 |
| `--input` | 输入文件路径 | data/input.json |
| `--output` | 输出文件路径 | data/output.json |
| `--chunk-trigger` | 触发分块的 token 阈值 | 800 |
| `--chunk-budget` | 每块 token 预算 | 600 |
| `--max-workers` | 并行 worker 数 | 3 |
| `--fast-mode` | 使用规则合并替代 LLM 合并 | false |
| `--skip-coref` | 跳过共指消解 | false |

## 军事领域抽取能力

军事 skill 覆盖以下关系类型：

- **兵力编成**：单位-职责、单位-部署位置、主防位置、推进轴线
- **指挥控制**：指挥关系、节制转换、通信/火控节点
- **火力对抗**：火力覆盖/打击对象、干扰压制对象
- **阶段约束**：阶段时间、失败条件、禁止事项
- **关键节点**：桥梁、道路、补给线的封锁/切断/保障
- **基础元信息**：文件等级、作战区域、行动代号、参战单位
- **建设部署**：发电厂、兵营、战车工厂等建设关系
- **敌军装备**：犀牛坦克、天启坦克、磁暴线圈等
- **弱点关系**：敌方依赖电力、心灵信标等
- **失败条件**：基地车损毁、铁幕装置启动等

## 输出格式

```json
{
  "relation_list": [
    {
      "head": "第一机步营",
      "relation": "集结于",
      "tail": "西侧盐盘集结区",
      "evidence": "第一机步营由西侧盐盘集结区沿峡谷主路北推",
      "skill": "military"
    }
  ]
}
```

## 扩展技能

在 `skills/` 目录下创建新的 JSON 文件即可添加领域技能：

```json
{
  "name": "your_skill",
  "description": "领域描述",
  "focus": "关注重点",
  "head_prior": "head 实体类型",
  "tail_prior": "tail 实体类型",
  "relation_style": "关系词风格",
  "negative_scope": "不应抽取的范围",
  "extraction_rules": ["规则1", "规则2"],
  "keywords": ["关键词1", "关键词2"],
  "fewshot": [
    {
      "text": "示例文本",
      "json": "{\"relation_list\":[...]}",
      "is_document_level": true
    }
  ]
}
```

## 设计原则

- 技能本体与执行逻辑解耦
- 通过 `skills/*.json` 扩展技能集
- 路由、抽取、整理可单独迭代
- 输出结构与现有实验兼容

## 许可证

MIT License
