from app.pipeline import _build_relation_batches, _build_relation_sections, _strip_markdown_image_content


def relation_config(split_mode, **changes):
    return {"split_mode": split_mode, "batch_size": 1, "max_batch_tokens": 2500,
            "include_parent_title": True, **changes}


def sample_document():
    return "一、第一章\n\n1.1 第一节\n\n甲单位抵达。\n\n1.2 第二节\n\n乙单位支援。\n\n二、第二章\n\n2.1 第三节\n\n丙单位保障。"


def test_strip_image_markup_preserves_surrounding_text():
    text = '甲方签署协议。\n\n![扫描件](data:image/png;base64,AAAA)\n\n<img src="attachment:image-1.png" />\n\n![流程图][flow]\n\n[flow]: images/flow.png\n\n乙方负责交付。'
    assert _strip_markdown_image_content(text) == "甲方签署协议。\n\n乙方负责交付。"


def test_small_section_keeps_parent_context():
    batches = _build_relation_batches(_build_relation_sections(sample_document()), relation_config("small_section"))
    assert [item["section_ids"] for item in batches] == [["1.1"], ["1.2"], ["2.1"]]
    assert batches[0]["text"].startswith("一、第一章\n\n1.1 第一节")


def test_markdown_sections_keep_document_title():
    text = "## 《文档标题》\n\n文件说明。\n\n## 一、第一章\n\n### 1.1 第一节\n\n甲单位抵达。\n\n### 1.2 第二节\n\n乙单位支援。"
    sections = _build_relation_sections(text)
    assert [item["section_id"] for item in sections] == ["doc-title", "1.1", "1.2"]
    assert sections[1]["parent_title"] == "一、第一章"


def test_chapter_batching_preserves_section_coverage():
    sections = _build_relation_sections(sample_document())
    batches = _build_relation_batches(sections, relation_config("chapter", chapter_batch_size=1))
    assert [item["section_ids"] for item in batches] == [["1.1", "1.2"], ["2.1"]]
    combined = _build_relation_batches(sections, relation_config("chapter", chapter_batch_size=2))
    assert [item["section_ids"] for item in combined] == [["1.1", "1.2", "2.1"]]


def test_paragraph_batching_keeps_section_attribution():
    text = "一、第一章\n\n1.1 第一节\n\n甲单位抵达。\n\n乙单位支援。\n\n丙单位保障。\n\n1.2 第二节\n\n丁单位控制。"
    batches = _build_relation_batches(_build_relation_sections(text), relation_config("paragraph", paragraph_batch_size=2))
    assert [item["section_ids"] for item in batches] == [["1.1"], ["1.1", "1.2"]]


def test_fixed_sections_count():
    batches = _build_relation_batches(_build_relation_sections(sample_document()), relation_config("fixed_sections", fixed_section_batch_size=2))
    assert [item["section_ids"] for item in batches] == [["1.1", "1.2"], ["2.1"]]


def test_long_section_respects_token_limit():
    text = "一、第一章\n\n1.1 第一节\n\n" + "甲" * 80 + "\n\n1.2 第二节\n\n乙方抵达。"
    batches = _build_relation_batches(_build_relation_sections(text), relation_config("chapter", max_batch_tokens=10))
    assert all(item["estimated_tokens"] <= 10 for item in batches)
    assert any(item["split_mode"] == "hard_split" for item in batches)
    assert any(item["section_ids"] == ["1.2"] for item in batches)
