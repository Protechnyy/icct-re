# 抽取配置与结果

抽取配置管理实体类型、关系定义及每个关系允许的头尾类型组合。完整默认配置位于 `skill4re/skill4re/default_extraction_schema.json`，其中的示例与运行配置使用相同的校验规则。

## 配置字段

- `schema_version` 表示配置文件格式，当前为 1；`revision` 表示每次成功保存的版本。
- `entity_types` 保存 `id`、`name`、`description`、`parent_id` 和 `examples`。ID 创建后保持不变，名称可以修改。
- `relation_types` 保存 `id`、`name`、`description`、`enabled`、`symmetric`、`allowed_pairs` 和 `examples`。
- `allowed_pairs` 的每一行是一组 `head_type_id` 与 `tail_type_id`，同一行的两个类型共同决定允许组合。
- 实体类型示例使用 `text`、`entity_name`、`type_id`、`evidence`；关系示例使用 `text`、`head`、`tail`、`head_type_id`、`tail_type_id`、`evidence`。证据必须出现在示例原文中。

关系引用父类型时，允许该类型及其全部后代类型。实体识别范围是所有启用关系涉及的类型及其后代类型；未参与关系的符合范围实体也会保留。

## 默认关系

| 关系 | 方向 | 允许组合 |
| --- | --- | --- |
| 指挥 | 指挥方指向被指挥方 | 指挥机构与指挥机构、前出分队或武器平台 |
| 协同 | 对称 | 指挥机构与指挥机构；武器平台与武器平台 |
| 情报保障 | 情报提供方指向接收方 | 侦查平台与指挥机构 |

武器平台的子类型为水面打击力量、水下打击力量、空中打击力量、长导打击力量。子类型之间可以按武器平台的组合规则建立协同关系。关系成立还需要原文明确证据。

## 结果字段

结果格式为 `result_schema_version=4`，完整示例见 [typed-result.json](examples/typed-result.json)。

- 实体：`entity_id`、`name`、`type_ids`、`aliases`。别名保存 `name`。
- 关系：`relation_id`、`relation_type_id`、`head_id`、`tail_id`。

原文引用属于抽取期间的上下文，用于判断实体身份与校验类型和关系。抽取服务完成校验后返回业务结构，结果读取、保存与导出直接校验该结构。JSON 保留完整文档用于预览，实体与关系记录使用上述字段。版本 2、版本 3 与无版本的历史结果在读取时转换，历史文件保持原样。

实体表展示名称、别名和类型。关系表展示头实体、关系、尾实体及两端类型。实体 CSV 每个实体一行，包含 ID、名称、类型和别名；关系 CSV 包含关系及两端实体的 ID、名称和类型。
