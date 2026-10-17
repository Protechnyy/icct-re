# 验证记录

日期：2026-10-17。

## 配置与接口检查

- 32 项检查通过，覆盖公共环境变量、校验运行参数、停用名称提示、无效配置、设置文件、不可变快照、并发原子保存、文档客户端、请求超时和文档工具。
- 真实模型设置 GET、PUT 已检查模型切换、模型名称恢复默认、空白 Key 保留和设置读取。健康检查返回 `status=ok`，Redis、OCR 和公共模型服务均可访问。
- 两个真实请求分别使用 23 秒公共请求超时，以及约 10 秒的剩余校验时限，均正常返回。
- 前端构建、Python 编译检查、`pip check`、`git diff --check` 和 OpenSpec strict 检查通过。

检查命令：

```bash
TMPDIR="$PWD/data/shared_llm_browser/runtime" PYTHONPATH=backend backend/.venv/bin/python -m pytest backend/tests/test_shared_llm_configuration.py backend/tests/test_agent_config.py backend/tests/test_skill4re_client.py backend/tests/test_agent_document.py backend/tests/test_utils.py -q -p no:cacheprovider
```

## 文档配置快照

使用真实 PDF 的首页，向隔离的 API 连续提交五个任务，由两个真实 Worker、Redis、PaddleOCR 和远程模型处理。五次提交使用相同原文，检查模型配置的生效时间和请求归属。

| task_id | 模型 | batch 数量 | 关系数量 | 校验状态 |
| --- | --- | --- | --- | --- |
| `c9487be5940d4afbab6776ebf7467118` | `qwen3.8-27b` | 2 | 33 | `partial` |
| `10561bc1554d40d0a2a0f05a69092f0d` | `qwen3.8-flash` | 2 | 36 | `partial` |
| `c216c4af7f824d81a0cf7e7261ea5677` | `qwen3.8-flash` | 2 | 48 | `partial` |
| `a28ec7b0913543fd8357c1191ef03b36` | `qwen3.8-27b` | 2 | 36 | `partial` |
| `2f9f1041484e4fd087642ddf42f91d0c` | `qwen3.8-27b` | 2 | 36 | `disabled` |

- 首个任务在 OCR 期间保存 `qwen3.8-flash`。其路由、全部 batch 和校验继续使用 `qwen3.8-27b`；已经排队的第二个任务开始处理时使用新模型。
- 第三个任务进入抽取后保存 `qwen3.8-27b`。第三个任务继续使用 `qwen3.8-flash`，随后开始的第四个任务使用 `qwen3.8-27b`。
- Worker 处理时间重叠，各文档保持自身配置；四个启用校验的任务均执行了实际文档工具调用。
- 第五个任务指定 `fast_mode=true`，校验状态为 `disabled`。
- 四次校验的 `partial` 状态保留在结果和轨迹中。本次记录配置行为和真实调用结果，关系质量指标没有纳入本次检查。

结果与实际请求日志位于 `data/shared_llm_live/`，入口为 `live-check.json`、`worker-1.log`、`worker-2.log` 和 `results/<task_id>/result.json`。原始 PDF 内容保持不变。

## 独立入口与浏览器

- `probe-agent-connection.py` 使用真实文档的 `read_section` 工具，模型确实执行查询并完成回答。
- `replay-agent-verification.py` 使用公共配置完成重放，输入文件 SHA256 保持不变。
- `run-agent-document-evaluation.py` 完成真实 OCR、抽取和校验重放，任务为 `shared-llm-evaluation-29b487c597cd43deb688b7dd8ecb99ce`，校验状态为 `completed`。该次独立入口检查使用 `VERIFICATION_AGENT_MAX_TASKS=2`、`VERIFICATION_AGENT_MAX_STEPS=10`。
- 独立评估的公共默认模型为 `qwen3.8-27b`，保存模型为 `qwen3.8-flash`；指定独立结果输出目录后，任务 `shared-llm-saved-settings-d43ab93d10874106860d85fee1570275` 的实际路由、抽取和校验均使用 `qwen3.8-flash`。该项检查关闭 streaming thinking，采用两项校验任务及十个执行步骤的运行限制。
- requests 路由、抽取和 coref 调用使用真实公共模型连接完成；SDK 调用由上述文档任务覆盖。
- 真实浏览器在 1366×900 和 320×844 下通过 DOM 与键盘检查：打开弹窗、公共配置说明、保存、重新读取、Escape 关闭和焦点恢复。弹窗处于窗口范围内，页面没有 JavaScript 错误。
- 24 份日志、结果及验证报告通过 API Key 内容检查；受限权限的模型设置文件保留认证信息。

对应材料位于 `data/shared_llm_entrypoints/entrypoints.json`、`data/shared_llm_timeout/timeouts.json`、`data/shared_llm_ui/ui-check.json` 和 `data/shared_llm_live/privacy-check.json`。

独立输出目录检查材料位于 `data/shared_llm_live/evaluation-settings-check.json`。浏览器检查使用已安装的 Chrome 和 Playwright；Playwright 安装在被 Git 忽略的检查目录：`npm install --prefix data/shared_llm_browser playwright --no-package-lock`。

## 部署配置

`backend/.env.example` 使用公共 `LLM_*` 和校验运行 `VERIFICATION_AGENT_*`。已有 `settings/model.json` 保持原结构。真实检查在独立存储和进程中使用已准备的公共配置，本机 `backend/.env` 保留原内容；重新启动日常服务前需要按照 README 的迁移表更新名称。
