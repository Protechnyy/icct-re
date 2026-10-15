# Proposal

## Why

当前点击顶部 `Qwen / vLLM` 会切换到独立的模型设置页面，离开正在使用的任务中心或 Skills 页面。API Key 输入框长期留空，同时承担保留和更换 Key 两种操作，需要更明确的配置状态、编辑方式和保存反馈。

## What Changes

- 顶部 `Qwen / vLLM` 入口采用与 `Skills` 一致的机器人图标文字按钮，统一字号、图标间距及悬停效果，保留与服务状态之间的分隔线。
- 点击顶部按钮打开当前页面上的模型设置弹窗；关闭后继续使用原页面。
- 继续维护一组 API 请求地址、Key 和模型名称；模型名称默认显示当前配置，用户可以编辑并保存自定义名称。
- 已配置 Key 时提供“更换 Key”操作；新 Key 使用密码输入框，取消更换或留空会保留现有 Key。
- 参考 CCswitch 的表单布局展示 API 请求地址、API Key 和模型名称，并整理字段说明、状态和操作按钮；提供字段校验、读取重试和保存错误提示。
- 留空模型名称时使用当前后端默认模型；保存自定义名称后，后续主模型请求和健康检查使用该名称。
- 显式点击“保存”后更新配置；成功后关闭弹窗并刷新服务状态，失败时保留输入内容。
- 处理未保存修改的关闭确认、保存期间的重复操作、键盘焦点和窄屏布局。

## Capabilities

### New Capabilities

- `model-api-settings`: 从顶部入口编辑单组模型 API 请求地址、Key 和模型名称，包括弹窗交互、默认值、自定义模型名称、Key 更换、保存与错误反馈。

### Modified Capabilities

无。

## Impact

- 前端：`frontend/src/App.jsx`、`frontend/src/components/ModelSettings.jsx` 和 `frontend/src/index.css`。
- 接口：扩展 `frontend/src/lib/api.js` 使用的 `GET /api/settings/model`、`PUT /api/settings/model`，读取默认及当前生效的模型名称，并提交模型名称修改。
- 后端：扩展单组配置存储及 `VllmClient`、`Skill4ReClient` 主模型请求，支持自定义模型名称和恢复默认；继续保持 Key 不回传、Key 留空保留现有值。
- 依赖：复用项目已有的 Arco Design Modal、Input、Alert 和 Button。
- 验证：后端配置/API/模型请求检查、前端构建，以及连接真实本地服务的 DOM、键盘操作和配置保存检查。
