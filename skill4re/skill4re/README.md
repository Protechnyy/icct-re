# Skill4RE

`skill4re` 是从 `examples/run_fight_skill_router_trial.py` 拆出来的可迭代项目版本，用于军事文档关系抽取。

## 目录

- `skills/`: skill 定义，使用独立 JSON 文件管理
- `loader.py`: skill 加载
- `backends.py`: API / 本地模型后端
- `routing.py`: 关键词先验、LLM 路由、缓存
- `prompts.py`: router / extraction / refinement prompts
- `normalization.py`: 清洗、去重、并列展开
- `service.py`: 关系抽取服务编排
- `run.py`: CLI 入口

当前抽取流程：

1. 先做文档级 skill 路由。
2. 长文触发切块后，每个 chunk 会再做一次轻量 skill 路由，并与文档级路由结果合并。
3. 各 chunk 使用自己的 skill 组合抽取候选关系。
4. extraction prompt 使用短通用规则，并按当前选中的 skill 动态注入专项规则。
5. 后处理会根据运行时加载的 skill 集校验 `skill` 字段，不再硬编码固定 skill 名称。
6. 有原文上下文时，后处理会过滤证据无法定位、且 head/tail 也无法同时在原文中找到的关系。

当前结果后处理分两层：

- `refinement`: 轻量整理、去重、规范化
- `domain reflection`: 面向军事文档的保守型领域反思，只做补漏和纠偏，不做大规模删减

## 运行

在仓库根目录执行：

```bash
python skill4re/run.py --backend qwen_api --model qwen3-32b --limit 1
```

也可以使用 OpenAI 官方 API：

```bash
python skill4re/run.py --backend openai --model gpt-4o-mini --limit 1
```

首条样本写入 `data_example` 的示例：

```bash
python skill4re/run.py \
  --backend qwen_api \
  --model qwen3-32b \
  --limit 1 \
  --output /home/users/lhy/OneKE/data_example/fight_data_skill4re_first.json
```

## 设计原则

- skill 本体与执行逻辑解耦
- 可以通过增加 `skills/*.json` 扩展 skill 集
- 路由、抽取、整理保持可单独迭代
- 输出结构保持与现有实验兼容
