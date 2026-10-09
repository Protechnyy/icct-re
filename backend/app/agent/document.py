"""In-memory paragraph BM25 and evidence locations for a single document."""
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
import math
import re


def compact(text):
    return re.sub(r"\s+", "", str(text or ""))


def bigrams(text):
    value = compact(text).lower()
    return [value[i:i + 2] for i in range(len(value) - 1)] if len(value) > 1 else list(value)


class DocumentIndex:
    def __init__(self, sections, document_text):
        self.sections = {str(section["section_id"]): deepcopy(section) for section in sections}
        self.document_text = document_text
        self.paragraphs = []
        for section_id, section in self.sections.items():
            paragraphs = section.get("paragraphs") or [{"content": section.get("text", ""),
                                                       "page": section.get("page_start")}]
            for position, paragraph in enumerate(paragraphs):
                if paragraph.get("content"):
                    self.paragraphs.append({"section_id": section_id, "paragraph_index": position,
                        "page": paragraph.get("page"), "content": paragraph["content"],
                        "global_block_id": paragraph.get("global_block_id"), "bbox": paragraph.get("bbox")})
        self.tokens = [Counter(bigrams(paragraph["content"])) for paragraph in self.paragraphs]
        self.df = Counter(token for tokens in self.tokens for token in tokens)
        self.average_length = sum(sum(tokens.values()) for tokens in self.tokens) / max(1, len(self.tokens))

    def search(self, query, limit=5):
        query_tokens = set(bigrams(query))
        ranked = []
        count = len(self.tokens)
        for index, tokens in enumerate(self.tokens):
            score = 0.0
            length = sum(tokens.values())
            for token in query_tokens & tokens.keys():
                frequency = tokens[token]
                idf = math.log(1 + (count - self.df[token] + 0.5) / (self.df[token] + 0.5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (
                    0.25 + 0.75 * length / max(1, self.average_length)))
            if score > 0:
                ranked.append({**self.paragraphs[index], "score": round(score, 6)})
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:max(0, min(int(limit), 5))]

    def search_document(self, query):
        results = self.search(query)
        return {"matches": results, "message": "" if results else "没有命中"}

    def read_section(self, section_id, max_chars=4000):
        section = self.sections.get(str(section_id))
        if section is None:
            return {"error": "小节不存在", "available_section_ids": list(self.sections)}
        text = section.get("text") or "\n\n".join(p.get("content", "") for p in section.get("paragraphs", []))
        return {"section_id": str(section_id), "title": section.get("title", ""),
                "text": text[:max_chars], "page_start": section.get("page_start"),
                "page_end": section.get("page_end"), "truncated": len(text) > max_chars}

    def locate_evidence(self, quote):
        needle = compact(quote)
        # Every write uses this exact whitespace-only check against the original document.
        if not needle or needle not in compact(self.document_text):
            candidates = []
            for paragraph in self.paragraphs:
                for sentence in re.split(r"(?<=[。！？.!?])|\n", paragraph["content"]):
                    if sentence.strip():
                        score = SequenceMatcher(None, needle, compact(sentence)).ratio()
                        candidates.append({**paragraph, "content": sentence.strip(), "similarity": score})
            closest = sorted(candidates, key=lambda item: item["similarity"], reverse=True)[:3]
            return {"matched": False, "locations": [], "closest_sentences": closest}
        locations = []
        for section_id in self.sections:
            paragraphs = [p for p in self.paragraphs if p["section_id"] == section_id]
            text = "".join(compact(p["content"]) for p in paragraphs)
            offset = 0
            while True:
                start = text.find(needle, offset)
                if start < 0:
                    break
                end = start + len(needle)
                cursor = 0
                for paragraph in paragraphs:
                    stop = cursor + len(compact(paragraph["content"]))
                    if cursor < end and stop > start:
                        location = deepcopy(paragraph)
                        if location not in locations:
                            locations.append(location)
                    cursor = stop
                offset = start + 1
        # Text can exist outside indexed sections (e.g. a legacy result). Never invent a page.
        if not locations:
            locations.append({"section_id": None, "page": None, "content": quote,
                              "global_block_id": None, "bbox": None})
        return {"matched": True, "locations": locations, "closest_sentences": []}


def query_entity(relations, aliases, name):
    canonical = next((key for key, values in aliases.items() if name == key or name in values), name)
    related = [deepcopy(item) for item in relations if canonical in (item.get("head"), item.get("tail"))]
    names = {str(item[field]) for item in relations for field in ("head", "tail") if item.get(field)}
    similar = sorted(candidate for candidate in names if candidate != name and (
        name in candidate or candidate in name or SequenceMatcher(None, name, candidate).ratio() >= 0.65)) if name else []
    return {"found": bool(related), "name": canonical, "aliases": list(aliases.get(canonical, [])),
            "section_ids": sorted({str(section) for item in related for section in item.get("source_sections", [])}),
            "relations": related, "similar_names": similar}
