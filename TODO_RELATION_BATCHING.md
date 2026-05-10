# 关系抽取分批优化 TODO

当前已支持按小节、大章、段落或固定 N 小节切分关系抽取输入。后续优化重点如下。

## 待优化项

- 增加更准确的 tokenizer 级 token 估算，替换当前字符长度近似值。
- 在 `chapter` 模式下，当大章过长时更精细地降级到小节或段落。
- 结果中的 `source_sections` 已保留，后续可结合 `ocr_paragraphs.json` 进一步回查页码和 bbox。
- 前端可继续补充说明性 tooltip，但不要增加大段使用说明。
- 补充端到端测试，覆盖上传参数到 Worker payload 再到最终结果的完整链路。

## 默认建议

```env
RELATION_SPLIT_MODE=small_section
RELATION_BATCH_SIZE=1
RELATION_MAX_BATCH_TOKENS=2500
RELATION_INCLUDE_PARENT_TITLE=true
```
