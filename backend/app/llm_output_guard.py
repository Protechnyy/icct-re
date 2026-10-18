from collections import Counter

from pydantic_core import from_json


class OutputRepetitionGuard:
    def __init__(self):
        self.parts = []
        self.length = 0
        self.next_check = 1024
        self.trailing_whitespace = 0

    def feed(self, text):
        self.parts.append(text)
        self.length += len(text)
        self.trailing_whitespace = self.trailing_whitespace + len(text) if not text.strip() else len(text) - len(text.rstrip())
        if self.trailing_whitespace >= 1024:
            return "模型持续输出空白字符，已终止请求"
        if self.length < self.next_check:
            return None
        self.next_check = self.length + 1024
        payload = from_json("".join(self.parts), allow_partial=True, cache_strings=False)
        pending = [payload]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                references = [item["source_id"] for item in value if isinstance(item, dict) and set(item) == {"source_id"}]
                if references and Counter(references).most_common(1)[0][1] >= 4:
                    return "模型在同一证据列表中重复来源编号，已终止请求"
                pending.extend(value)
        if not isinstance(payload, dict):
            return None
        relations = Counter()
        for relation in payload.get("relations", []):
            if not isinstance(relation, dict) or not {"relation_type_id", "head_id", "tail_id", "evidence"} <= set(relation):
                continue
            evidence = relation["evidence"]
            if not evidence or any(not isinstance(item, dict) or set(item) != {"source_id"} for item in evidence):
                continue
            key = (relation["relation_type_id"], relation["head_id"], relation["tail_id"], tuple(sorted(item["source_id"] for item in evidence)))
            relations[key] += 1
            if relations[key] >= 4:
                return "模型重复同一关系及证据，已终止请求"
        return None
