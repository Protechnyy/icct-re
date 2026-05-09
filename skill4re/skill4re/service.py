from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from skill4re.backends import (
    RELATION_LIST_RESPONSE_FORMAT,
    LocalQwenGenerator,
    generate_text,
    generate_text_with_requests,
)
from skill4re.coref import resolve_coreferences
from skill4re.dataset import approx_token_count, chunk_document
from skill4re.normalization import merge_chunk_relations, normalize_prediction, split_by_confidence
from skill4re.parsing import parse_json
from skill4re.prompts import (
    build_extraction_prompt,
    build_proofreading_prompt,
    build_summarize_prompt,
    build_targeted_proofreading_prompt,
)
from skill4re.routing import keyword_scores, route_document
from skill4re.models import Skill
from skill4re.types import ExtractionResult


logger = logging.getLogger(__name__)


class SkillRouterExtractor:
    CHUNK_REROUTE_MIN_SCORE = 2
    CHUNK_REROUTE_SCORE_MARGIN = 2

    def __init__(
        self,
        skills: List[Skill],
        backend: str,
        model: str,
        client: Optional[OpenAI],
        local_generator: Optional[LocalQwenGenerator],
        route_cache: Dict,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: int = 120,
        enable_thinking: bool = False,
        fast_mode: bool = False,
        skip_coref: bool = False,
    ):
        self.skills = skills
        self.backend = backend
        self.model = model
        self.client = client
        self.local_generator = local_generator
        self.route_cache = route_cache
        self.api_key = api_key
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.enable_thinking = enable_thinking
        self.fast_mode = fast_mode
        self.skip_coref = skip_coref
        self.skill_by_name = {skill.name: skill for skill in skills}
        self.valid_skill_names = set(self.skill_by_name)

    def _call_llm(self, prompt: str, max_tokens: int) -> Dict:
        """统一的 LLM 调用接口。

        根据 backend 类型选择调用方式，解析 JSON 结果。
        """
        if self.client is not None or self.local_generator is not None:
            content = generate_text(
                prompt=prompt,
                backend=self.backend,
                api_client=self.client,
                local_generator=self.local_generator,
                model=self.model,
                max_tokens=max_tokens,
                response_format=RELATION_LIST_RESPONSE_FORMAT,
            )
        else:
            content = generate_text_with_requests(
                prompt=prompt,
                api_key=self.api_key,
                model=self.model,
                max_tokens=max_tokens,
                backend=self.backend,
                response_format=RELATION_LIST_RESPONSE_FORMAT,
                base_url=self.base_url,
                timeout=self.request_timeout,
                enable_thinking=self.enable_thinking,
            )
        return parse_json(content)

    def _get_max_tokens(self, task_type: str) -> int:
        """根据任务类型获取 max_tokens 配置。"""
        token_configs = {
            "extraction": {"local_qwen3": 800, "qwen_api": 2200, "vllm": 1800, "default": 4200},
            "proofreading": {"local_qwen3": 1400, "qwen_api": 2600, "vllm": 1600, "default": 3600},
            "summarize": {"local_qwen3": 2000, "qwen_api": 3000, "vllm": 1800, "default": 4000},
            "targeted_proofread": {"local_qwen3": 1000, "qwen_api": 1800, "vllm": 1200, "default": 2400},
        }
        config = token_configs.get(task_type, token_configs["extraction"])
        return config.get(self.backend, config["default"])

    def should_reroute_chunk(
        self,
        chunk_scores: Dict[str, int],
        document_selected_skills: List[Skill],
    ) -> bool:
        document_skill_names = {skill.name for skill in document_selected_skills}
        if len(document_skill_names) >= len(self.skills):
            return False

        document_best_score = max(
            (chunk_scores.get(skill.name, 0) for skill in document_selected_skills),
            default=0,
        )
        outside_best_score = max(
            (
                score
                for skill_name, score in chunk_scores.items()
                if skill_name not in document_skill_names
            ),
            default=0,
        )
        return (
            outside_best_score >= self.CHUNK_REROUTE_MIN_SCORE
            and outside_best_score
            >= document_best_score + self.CHUNK_REROUTE_SCORE_MARGIN
        )

    def merge_skill_orders(self, *skill_groups: List[Skill], max_skills: int = 3) -> List[Skill]:
        ordered: List[Skill] = []
        seen = set()
        for group in skill_groups:
            for skill in group:
                if skill.name in seen:
                    continue
                ordered.append(skill)
                seen.add(skill.name)
                if len(ordered) >= max_skills:
                    return ordered
        return ordered

    def extract_chunk(self, chunk_index: int, chunk_text: str, selected_skills: List[Skill]) -> Dict:
        started_at = time.perf_counter()
        prompt = build_extraction_prompt(chunk_text, selected_skills)
        max_tokens = self._get_max_tokens("extraction")
        parsed_chunk = self._call_llm(prompt, max_tokens)
        return {
            "chunk_index": chunk_index,
            "chunk_token_count": approx_token_count(chunk_text),
            "selected_skills": [skill.name for skill in selected_skills],
            "elapsed_seconds": round(time.perf_counter() - started_at, 4),
            "prediction": parsed_chunk,
        }

    def proofread_prediction(
        self,
        doc: str,
        selected_skills: List[Skill],
        prediction: Dict,
    ) -> Tuple[Dict, float, Dict]:
        relation_list = prediction.get("relation_list", [])
        if not relation_list:
            return prediction, 0.0, {"input_relation_count": 0, "output_relation_count": 0}
        prompt = build_proofreading_prompt(doc, selected_skills, relation_list)
        started_at = time.perf_counter()
        try:
            max_tokens = self._get_max_tokens("proofreading")
            proofread = self._call_llm(prompt, max_tokens)
            merged = merge_chunk_relations(
                [prediction, proofread],
                valid_skill_names=self.valid_skill_names,
                source_text=doc,
            )
            elapsed = round(time.perf_counter() - started_at, 4)
            return merged, elapsed, {
                "input_relation_count": len(relation_list),
                "output_relation_count": len(merged.get("relation_list", [])),
            }
        except Exception as exc:
            logger.warning(
                "Proofreading failed; falling back to normalized prediction: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            elapsed = round(time.perf_counter() - started_at, 4)
            normalized = normalize_prediction(
                prediction,
                valid_skill_names=self.valid_skill_names,
                source_text=doc,
            )
            return normalized, elapsed, {
                "input_relation_count": len(relation_list),
                "output_relation_count": len(normalized.get("relation_list", [])),
            }

    def summarize_chunks(
        self,
        doc: str,
        selected_skills: List[Skill],
        chunk_predictions: List[Dict],
    ) -> Tuple[Dict, float]:
        """语义合并多个 chunk 的抽取结果。

        与 merge_chunk_relations（纯规则去重）不同，这里调用 LLM 做跨 chunk 的
        实体归一化和语义级去重。仅在 chunk 数量 >= 2 时触发。

        fast_mode=True 时使用规则合并，不调用 LLM。
        """
        if len(chunk_predictions) < 2:
            return chunk_predictions[0] if chunk_predictions else {"relation_list": []}, 0.0

        # fast_mode: 使用规则合并，不调用 LLM
        if self.fast_mode:
            started_at = time.perf_counter()
            fallback = merge_chunk_relations(
                [item["prediction"] for item in chunk_predictions],
                valid_skill_names=self.valid_skill_names,
                source_text=doc,
            )
            elapsed = round(time.perf_counter() - started_at, 4)
            logger.info("Fast mode: rule-based merge for %d chunks", len(chunk_predictions))
            return fallback, elapsed

        # 只传 relation_list 给 LLM
        chunk_rels = [
            {"chunk_index": item["chunk_index"], "relation_list": item["prediction"].get("relation_list", [])}
            for item in chunk_predictions
        ]
        prompt = build_summarize_prompt(doc, selected_skills, chunk_rels)
        started_at = time.perf_counter()
        try:
            max_tokens = self._get_max_tokens("summarize")
            summarized = self._call_llm(prompt, max_tokens)
            elapsed = round(time.perf_counter() - started_at, 4)
            logger.info("Summarize done: %d chunks merged", len(chunk_predictions))
            return summarized, elapsed
        except Exception as exc:
            logger.warning(
                "Summarize failed; falling back to rule-based merge: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            elapsed = round(time.perf_counter() - started_at, 4)
            fallback = merge_chunk_relations(
                [item["prediction"] for item in chunk_predictions],
                valid_skill_names=self.valid_skill_names,
            )
            return fallback, elapsed

    def targeted_proofread(
        self,
        doc: str,
        selected_skills: List[Skill],
        prediction: Dict,
    ) -> Tuple[Dict, float, Dict]:
        """低置信度关系反思。

        将关系分为高/低置信度两组，高置信度直接保留，只对低置信度的做 LLM 反思。
        比通用 proofreading 更有针对性，也更快（输入更少）。
        """
        relation_list = prediction.get("relation_list", [])
        if not relation_list:
            return prediction, 0.0, {
                "input_relation_count": 0,
                "high_confidence_count": 0,
                "low_confidence_count": 0,
                "output_relation_count": 0,
            }

        high_conf, low_conf = split_by_confidence(relation_list, source_text=doc)

        # 如果没有低置信度的，跳过反思
        if not low_conf:
            logger.info("Targeted proofread skipped: all %d relations are high confidence", len(high_conf))
            return prediction, 0.0, {
                "input_relation_count": len(relation_list),
                "high_confidence_count": len(high_conf),
                "low_confidence_count": 0,
                "output_relation_count": len(relation_list),
            }

        # 只对低置信度关系做反思
        prompt = build_targeted_proofreading_prompt(doc, selected_skills, low_conf)
        started_at = time.perf_counter()
        try:
            max_tokens = self._get_max_tokens("targeted_proofread")
            corrected = self._call_llm(prompt, max_tokens)
            corrected_rels = corrected.get("relation_list", [])
            # 合并：高置信度 + 修正后的低置信度
            merged = merge_chunk_relations(
                [{"relation_list": high_conf}, {"relation_list": corrected_rels}],
                valid_skill_names=self.valid_skill_names,
                source_text=doc,
            )
            elapsed = round(time.perf_counter() - started_at, 4)
            logger.info(
                "Targeted proofread done: %d high + %d low -> %d output",
                len(high_conf), len(low_conf), len(merged.get("relation_list", [])),
            )
            return merged, elapsed, {
                "input_relation_count": len(relation_list),
                "high_confidence_count": len(high_conf),
                "low_confidence_count": len(low_conf),
                "output_relation_count": len(merged.get("relation_list", [])),
            }
        except Exception as exc:
            logger.warning(
                "Targeted proofread failed; falling back to full proofread: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            # fallback: 用通用 proofreading
            return self.proofread_prediction(doc, selected_skills, prediction)

    def extract_document(
        self,
        doc_text: str,
        chunk_trigger: int,
        chunk_budget: int,
        max_workers: int,
    ) -> Dict:
        sample_started_at = time.perf_counter()
        routing_started_at = time.perf_counter()
        routing = route_document(
            backend=self.backend,
            client=self.client,
            local_generator=self.local_generator,
            text=doc_text,
            route_cache=self.route_cache,
            model=self.model,
            skills=self.skills,
            api_key=self.api_key,
            base_url=self.base_url,
            request_timeout=self.request_timeout,
            enable_thinking=self.enable_thinking,
        )
        routing_elapsed = round(time.perf_counter() - routing_started_at, 4)
        document_selected_skills = routing["selected_skills"]
        doc_tokens = approx_token_count(doc_text)
        used_chunking = doc_tokens > chunk_trigger

        # 共指消解：仅长文档触发，统一实体表述
        # skip_coref=True 时跳过共指消解
        coref_mapping: Dict[str, List[str]] = {}
        coref_elapsed = 0.0
        if used_chunking and not self.skip_coref:
            coref_started_at = time.perf_counter()
            doc_text, coref_mapping = resolve_coreferences(
                doc_text=doc_text,
                backend=self.backend,
                client=self.client,
                local_generator=self.local_generator,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                request_timeout=self.request_timeout,
                enable_thinking=self.enable_thinking,
            )
            coref_elapsed = round(time.perf_counter() - coref_started_at, 4)
        chunk_routes = []
        chunk_routing_elapsed = 0.0
        summarize_elapsed = 0.0
        if used_chunking:
            chunks = chunk_document(doc_text, chunk_budget)
            chunk_predictions = []
            worker_count = max(1, min(max_workers, len(chunks)))
            chunk_selected_skills = []
            for chunk_index, chunk_text in enumerate(chunks):
                chunk_routing_started_at = time.perf_counter()
                chunk_scores = keyword_scores(chunk_text, self.skills)
                if self.should_reroute_chunk(chunk_scores, document_selected_skills):
                    chunk_route = route_document(
                        backend=self.backend,
                        client=self.client,
                        local_generator=self.local_generator,
                        text=chunk_text,
                        route_cache=self.route_cache,
                        model=self.model,
                        skills=self.skills,
                        api_key=self.api_key,
                        base_url=self.base_url,
                        request_timeout=self.request_timeout,
                        enable_thinking=self.enable_thinking,
                    )
                    routing_mode = "chunk_rerouted"
                else:
                    chunk_route = {
                        "selected_skills": document_selected_skills,
                        "scores": chunk_scores,
                        "router_reason": "复用文档级路由；chunk 关键词分数未显示明显偏离。",
                        "router_mode": "document_reused",
                        "cache_hit": routing["cache_hit"],
                    }
                    routing_mode = "document_reused"
                chunk_routing_elapsed += time.perf_counter() - chunk_routing_started_at
                selected_for_chunk = self.merge_skill_orders(
                    chunk_route["selected_skills"],
                    document_selected_skills,
                )
                chunk_selected_skills.append(selected_for_chunk)
                chunk_routes.append(
                    {
                        "chunk_index": chunk_index,
                        "selected_skills": [skill.name for skill in selected_for_chunk],
                        "router_selected_skills": [
                            skill.name for skill in chunk_route["selected_skills"]
                        ],
                        "scores": chunk_route["scores"],
                        "router_reason": chunk_route["router_reason"],
                        "router_mode": chunk_route.get("router_mode", routing_mode),
                        "cache_hit": chunk_route["cache_hit"],
                        "routing_mode": routing_mode,
                    }
                )
            extraction_started_at = time.perf_counter()
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        self.extract_chunk,
                        chunk_index,
                        chunk_text,
                        chunk_selected_skills[chunk_index],
                    )
                    for chunk_index, chunk_text in enumerate(chunks)
                ]
                for future in as_completed(futures):
                    chunk_predictions.append(future.result())
            chunk_predictions.sort(key=lambda item: item["chunk_index"])
            normalized_chunk_predictions = [
                normalize_prediction(
                    item["prediction"],
                    valid_skill_names=self.valid_skill_names,
                    source_text=chunks[item["chunk_index"]],
                )
                for item in chunk_predictions
            ]
            # 语义合并：用 LLM 做跨 chunk 的实体归一化和关系去重
            parsed, summarize_elapsed = self.summarize_chunks(
                doc=doc_text,
                selected_skills=document_selected_skills,
                chunk_predictions=[
                    {"chunk_index": item["chunk_index"], "prediction": norm}
                    for item, norm in zip(chunk_predictions, normalized_chunk_predictions)
                ],
            )
            # 如果 summarize 失败（返回空），回退到规则合并
            if not parsed.get("relation_list"):
                parsed = merge_chunk_relations(
                    normalized_chunk_predictions,
                    valid_skill_names=self.valid_skill_names,
                )
            selected_skills = self.merge_skill_orders(
                document_selected_skills,
                *chunk_selected_skills,
                max_skills=len(self.skills),
            )
        else:
            chunks = [doc_text]
            chunk_predictions = []
            selected_skills = document_selected_skills
            extraction_started_at = time.perf_counter()
            single_result = self.extract_chunk(
                chunk_index=0,
                chunk_text=doc_text,
                selected_skills=selected_skills,
            )
            parsed = normalize_prediction(
                single_result["prediction"],
                valid_skill_names=self.valid_skill_names,
                source_text=doc_text,
            )
            chunk_predictions = [single_result]
        extraction_elapsed = round(time.perf_counter() - extraction_started_at, 4)
        if self.fast_mode:
            relation_count = len(parsed.get("relation_list", []))
            proofreading_elapsed = 0.0
            proofreading_summary = {
                "input_relation_count": relation_count,
                "high_confidence_count": relation_count,
                "low_confidence_count": 0,
                "output_relation_count": relation_count,
            }
        else:
            # 低置信度反思：只对低置信度关系做 LLM 审查
            parsed, proofreading_elapsed, proofreading_summary = self.targeted_proofread(
                doc=doc_text,
                selected_skills=selected_skills,
                prediction=parsed,
            )
        total_elapsed = round(time.perf_counter() - sample_started_at, 4)
        return {
            "preprocess": {
                "doc_token_count": doc_tokens,
                "used_chunking": used_chunking,
                "chunk_count": len(chunks),
                "chunk_trigger": chunk_trigger,
                "chunk_budget": chunk_budget,
                "coref_resolved": bool(coref_mapping),
                "coref_entity_groups": len(coref_mapping),
            },
            "routing": {
                "selected_skills": [skill.name for skill in selected_skills],
                "document_selected_skills": [skill.name for skill in document_selected_skills],
                "scores": routing["scores"],
                "router_reason": routing["router_reason"],
                "router_mode": routing.get("router_mode", "unknown"),
                "cache_hit": routing["cache_hit"],
                "chunk_routes": chunk_routes,
            },
            "timing": {
                "routing_seconds": routing_elapsed,
                "coref_seconds": coref_elapsed,
                "chunk_routing_seconds": round(chunk_routing_elapsed, 4),
                "extraction_seconds": extraction_elapsed,
                "summarize_seconds": round(summarize_elapsed, 4),
                "proofreading_seconds": proofreading_elapsed,
                "refinement_seconds": 0.0,
                "domain_reflection_seconds": 0.0,
                "total_seconds": total_elapsed,
            },
            "proofreading": proofreading_summary,
            "domain_reflection": proofreading_summary,
            "chunk_predictions": chunk_predictions,
            "prediction": parsed,
        }
