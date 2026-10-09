from app.agent.document import DocumentIndex, query_entity


def make_index():
    sections = [{"section_id": f"s{i}", "title": f"标题{i}", "page_start": i, "page_end": i,
        "text": f"第三营部署于河谷{i}。", "paragraphs": [{"page": i, "content": f"第三营部署于河谷{i}。"}]}
        for i in range(1, 8)]
    return DocumentIndex(sections, "\n\n".join(s["text"] for s in sections))


def test_bm25_returns_real_locations_and_no_hits():
    index = make_index()
    hits = index.search("河谷1")
    assert hits[0]["section_id"] == "s1" and hits[0]["page"] == 1
    assert index.search("完全无关的海洋") == []


def test_search_tool_has_hits_no_hits_and_limit():
    index = make_index()
    assert index.search_document("河谷1")["matches"]
    assert index.search_document("完全无关的海洋") == {"matches": [], "message": "没有命中"}
    assert len(index.search("第三营", limit=100)) == 5


def test_read_section_exists_missing_and_truncates():
    index = make_index()
    assert index.read_section("s2")["page_start"] == 2
    assert index.read_section("missing")["available_section_ids"] == list(index.sections)
    result = index.read_section("s1", max_chars=5)
    assert len(result["text"]) == 5 and result["truncated"]


def test_locate_exact_whitespace_and_miss():
    index = make_index()
    exact = index.locate_evidence("第三营部署于河谷1。")
    assert exact["matched"] and exact["locations"][0]["page"] == 1
    assert index.locate_evidence("第三营 部署于\n河谷1。") == exact
    missing = index.locate_evidence("第三营部署于海洋。")
    assert not missing["matched"] and missing["closest_sentences"]


def test_locate_all_hits_and_quote_spanning_paragraphs():
    sections = [{"section_id": "s1", "paragraphs": [
        {"content": "第三营部署于", "page": 1}, {"content": "河谷。", "page": 2}]}]
    index = DocumentIndex(sections, "第三营部署于\n河谷。")
    assert [p["page"] for p in index.locate_evidence("第三营部署于河谷。")["locations"]] == [1, 2]


def test_entity_known_similar_and_unknown():
    relations = [{"head": "第三营", "tail": "河谷", "source_sections": ["s1"]}]
    known = query_entity(relations, {"第三营": ["三营"]}, "三营")
    assert known["found"] and known["section_ids"] == ["s1"] and known["aliases"] == ["三营"]
    similar = query_entity(relations, {}, "第三")
    assert not similar["found"] and similar["similar_names"] == ["第三营"]
    unknown = query_entity(relations, {}, "完全未知")
    assert not unknown["found"] and not unknown["similar_names"]
