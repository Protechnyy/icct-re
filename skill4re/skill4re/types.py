"""Type definitions for skill4re extraction results."""

from typing import Dict, List, TypedDict


class PreprocessInfo(TypedDict):
    """预处理信息。"""
    doc_token_count: int
    used_chunking: bool
    chunk_count: int
    chunk_trigger: int
    chunk_budget: int
    coref_resolved: bool
    coref_entity_groups: int


class ChunkRoute(TypedDict):
    """Chunk 路由信息。"""
    chunk_index: int
    selected_skills: List[str]
    router_selected_skills: List[str]
    scores: Dict[str, float]
    router_reason: str
    router_mode: str
    cache_hit: bool
    routing_mode: str


class RoutingInfo(TypedDict):
    """路由信息。"""
    selected_skills: List[str]
    document_selected_skills: List[str]
    scores: Dict[str, float]
    router_reason: str
    router_mode: str
    cache_hit: bool
    chunk_routes: List[ChunkRoute]


class TimingInfo(TypedDict):
    """时间统计。"""
    routing_seconds: float
    coref_seconds: float
    chunk_routing_seconds: float
    extraction_seconds: float
    summarize_seconds: float
    proofreading_seconds: float
    refinement_seconds: float
    domain_reflection_seconds: float
    total_seconds: float


class ProofreadingSummary(TypedDict):
    """校对摘要。"""
    input_relation_count: int
    output_relation_count: int


class ChunkPrediction(TypedDict):
    """Chunk 抽取结果。"""
    chunk_index: int
    chunk_token_count: int
    selected_skills: List[str]
    elapsed_seconds: float
    prediction: Dict


class ExtractionResult(TypedDict):
    """完整的抽取结果。"""
    preprocess: PreprocessInfo
    routing: RoutingInfo
    timing: TimingInfo
    proofreading: ProofreadingSummary
    domain_reflection: ProofreadingSummary
    chunk_predictions: List[ChunkPrediction]
    prediction: Dict
