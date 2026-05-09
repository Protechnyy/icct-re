"""Normalization package for skill4re.

This package provides entity name fixing, evidence validation,
deduplication, and confidence scoring for relation extraction.
"""

# Entity name fixing
from skill4re.normalization.entity import fix_entity_name

# Evidence validation
from skill4re.normalization.evidence import (
    find_supporting_evidence,
    normalize_relation_text,
    relation_has_source_support,
    select_evidence,
    text_contains,
)

# Deduplication and merging
from skill4re.normalization.dedup import (
    compress_long_tail,
    expand_relation_item,
    fuzzy_entity_match,
    looks_like_bad_tail,
    looks_like_clause,
    merge_chunk_relations,
    normalize_prediction,
    normalize_relation_phrase,
    sanitize_relation_item,
    split_compound_tail,
)

# Confidence scoring
from skill4re.normalization.confidence import (
    HIGH_CONFIDENCE_RELATIONS,
    relation_confidence,
    split_by_confidence,
)

__all__ = [
    # Entity
    "fix_entity_name",
    # Evidence
    "find_supporting_evidence",
    "normalize_relation_text",
    "relation_has_source_support",
    "select_evidence",
    "text_contains",
    # Dedup
    "compress_long_tail",
    "expand_relation_item",
    "fuzzy_entity_match",
    "looks_like_bad_tail",
    "looks_like_clause",
    "merge_chunk_relations",
    "normalize_prediction",
    "normalize_relation_phrase",
    "sanitize_relation_item",
    "split_compound_tail",
    # Confidence
    "HIGH_CONFIDENCE_RELATIONS",
    "relation_confidence",
    "split_by_confidence",
]
