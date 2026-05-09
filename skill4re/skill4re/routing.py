import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from skill4re.backends import ROUTER_RESPONSE_FORMAT, LocalQwenGenerator, generate_text, generate_text_with_requests
from skill4re.models import Skill
from skill4re.parsing import parse_json
from skill4re.prompts import build_router_prompt


logger = logging.getLogger(__name__)
RULE_PRIMARY_MIN_SCORE = 1.0
RULE_AUX_MIN_SCORE = 1.0
RULE_CONFIDENT_TOP_SCORE = 3.0
RULE_CONFIDENT_MARGIN = 1.0


def compute_keyword_idf(skills: List[Skill]) -> Dict[str, float]:
    """计算每个关键词的 IDF 权重：只属于一个 skill 的关键词权重高，多个 skill 共有的权重低。"""
    keyword_to_skills: Dict[str, int] = {}
    for skill in skills:
        for kw in skill.keywords:
            if kw:
                keyword_to_skills[kw] = keyword_to_skills.get(kw, 0) + 1
    idf = {}
    n_skills = len(skills)
    for kw, count in keyword_to_skills.items():
        idf[kw] = math.log(n_skills / max(count, 1)) + 1.0  # +1 平滑
    return idf


def weighted_keyword_scores(text: str, skills: List[Skill]) -> Dict[str, float]:
    """IDF 加权的关键词分数：稀有关键词贡献大，常见关键词贡献小。"""
    idf = compute_keyword_idf(skills)
    scores = {}
    for skill in skills:
        score = 0.0
        for keyword in skill.keywords:
            if not keyword:
                continue
            count = text.count(keyword)
            weight = idf.get(keyword, 1.0)
            # 长关键词额外加权（军事术语通常是长词）
            if len(keyword) >= 4:
                weight *= 1.5
            score += count * weight
        scores[skill.name] = round(score, 2)
    return scores


def keyword_scores(text: str, skills: List[Skill]) -> Dict[str, float]:
    return weighted_keyword_scores(text, skills)


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_route_cache(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(
            "Failed to load route cache from %s: %s: %s",
            path,
            type(exc).__name__,
            exc,
        )
        return {}


def save_route_cache(path: Path, cache: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)


def select_skills_by_rules(
    scores: Dict[str, float],
    skills: List[Skill],
    max_skills: int,
) -> List[Skill]:
    ranked = sorted(skills, key=lambda item: scores[item.name], reverse=True)
    top_score = scores[ranked[0].name] if ranked else 0.0
    if top_score < RULE_PRIMARY_MIN_SCORE:
        return ranked[:max_skills]

    selected = [ranked[0]]
    for skill in ranked[1:]:
        if len(selected) >= max_skills:
            break
        if scores[skill.name] >= RULE_AUX_MIN_SCORE:
            selected.append(skill)
    return selected


def build_rule_reason(scores: Dict[str, float], selected: List[Skill]) -> str:
    selected_names = "、".join(skill.name for skill in selected)
    score_text = "，".join(f"{name}={score:g}" for name, score in scores.items())
    if not scores or max(scores.values(), default=0.0) < RULE_PRIMARY_MIN_SCORE:
        return f"规则路由：无明显关键词信号，为保证召回选择 {selected_names}。加权分数：{score_text}。"
    return f"规则路由：根据关键词加权分数选择 {selected_names}。加权分数：{score_text}。"


def can_run_llm_router(
    backend: str,
    client: Optional[OpenAI],
    local_generator: Optional[LocalQwenGenerator],
) -> bool:
    if backend == "local_qwen3":
        return local_generator is not None
    if backend == "vllm":
        return True
    if backend in {"openai", "api", "qwen_api"}:
        return client is not None
    return False


def should_use_llm_router(scores: Dict[str, float]) -> bool:
    ranked_scores = sorted(scores.values(), reverse=True)
    if not ranked_scores:
        return False
    top_score = ranked_scores[0]
    second_score = ranked_scores[1] if len(ranked_scores) > 1 else 0.0
    if top_score < RULE_PRIMARY_MIN_SCORE:
        return True
    if top_score < RULE_CONFIDENT_TOP_SCORE:
        return True
    return second_score >= RULE_AUX_MIN_SCORE and top_score - second_score < RULE_CONFIDENT_MARGIN


def cache_route_result(
    route_cache: Dict,
    cache_key: str,
    selected: List[Skill],
    scores: Dict[str, float],
    router_reason: str,
    router_mode: str,
) -> None:
    route_cache[cache_key] = {
        "selected_skills": [skill.name for skill in selected],
        "scores": scores,
        "router_reason": router_reason,
        "router_mode": router_mode,
    }


def build_result(
    selected: List[Skill],
    scores: Dict[str, float],
    router_reason: str,
    router_mode: str,
    cache_hit: bool,
) -> Dict:
    return {
        "selected_skills": selected,
        "scores": scores,
        "router_reason": router_reason,
        "router_mode": router_mode,
        "cache_hit": cache_hit,
    }


def route_with_llm(
    backend: str,
    client: Optional[OpenAI],
    local_generator: Optional[LocalQwenGenerator],
    text: str,
    model: str,
    skills: List[Skill],
    scores: Dict[str, float],
    valid_skill_names: set,
    max_skills: int,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    request_timeout: int = 120,
    enable_thinking: bool = False,
) -> Dict:
    prompt = build_router_prompt(text[:3000], scores, skills)
    if client is not None or local_generator is not None:
        content = generate_text(
            prompt=prompt,
            backend=backend,
            api_client=client,
            local_generator=local_generator,
            model=model,
            max_tokens=160 if backend == "local_qwen3" else 500,
            response_format=ROUTER_RESPONSE_FORMAT,
        )
    else:
        content = generate_text_with_requests(
            prompt=prompt,
            api_key=api_key,
            model=model,
            max_tokens=160 if backend == "local_qwen3" else 500,
            backend=backend,
            response_format=ROUTER_RESPONSE_FORMAT,
            base_url=base_url,
            timeout=request_timeout,
            enable_thinking=enable_thinking,
        )
    parsed = parse_json(content)
    ordered = []
    primary = parsed.get("primary_skill")
    if primary in valid_skill_names:
        ordered.append(primary)
    for name in parsed.get("aux_skills", []):
        if name in valid_skill_names and name not in ordered:
            ordered.append(name)
    if not ordered:
        raise ValueError("router returned empty skills")
    ordered = ordered[:max_skills]
    selected = [skill for skill in skills if skill.name in ordered]
    selected.sort(key=lambda item: ordered.index(item.name))
    return {
        "selected": selected,
        "router_reason": parsed.get("reason", ""),
    }


def route_document(
    backend: str,
    client: Optional[OpenAI],
    local_generator: Optional[LocalQwenGenerator],
    text: str,
    route_cache: Dict,
    model: str,
    skills: List[Skill],
    max_skills: int = 3,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    request_timeout: int = 120,
    enable_thinking: bool = False,
) -> Dict:
    scores = keyword_scores(text, skills)
    cache_key = doc_hash(text)
    cached = route_cache.get(cache_key)
    valid_skill_names = {skill.name for skill in skills}
    if isinstance(cached, dict):
        ordered = []
        for name in cached.get("selected_skills", []):
            if name in valid_skill_names and name not in ordered:
                ordered.append(name)
        if ordered:
            selected = [skill for skill in skills if skill.name in ordered]
            selected.sort(key=lambda item: ordered.index(item.name))
            return build_result(
                selected=selected,
                scores=cached.get("scores", scores),
                router_reason=cached.get("router_reason", "命中路由缓存。"),
                router_mode=cached.get("router_mode", "legacy_llm_cache"),
                cache_hit=True,
            )

    rule_selected = select_skills_by_rules(scores, skills, max_skills)
    rule_reason = build_rule_reason(scores, rule_selected)
    if should_use_llm_router(scores) and can_run_llm_router(backend, client, local_generator):
        try:
            llm_route = route_with_llm(
                backend=backend,
                client=client,
                local_generator=local_generator,
                text=text,
                model=model,
                skills=skills,
                scores=scores,
                valid_skill_names=valid_skill_names,
                max_skills=max_skills,
                api_key=api_key,
                base_url=base_url,
                request_timeout=request_timeout,
                enable_thinking=enable_thinking,
            )
            router_reason = llm_route["router_reason"]
            selected = llm_route["selected"]
            cache_route_result(route_cache, cache_key, selected, scores, router_reason, "llm_router")
            return build_result(selected, scores, router_reason, "llm_router", False)
        except Exception as exc:
            logger.warning(
                "LLM routing failed; falling back to keyword rules: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )

    cache_route_result(route_cache, cache_key, rule_selected, scores, rule_reason, "keyword_rule")
    return build_result(rule_selected, scores, rule_reason, "keyword_rule", False)
