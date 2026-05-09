import argparse
import json
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNK_BUDGET = 900
DEFAULT_CHUNK_TRIGGER = 1200
DEFAULT_MAX_WORKERS = 4
DEFAULT_LOCAL_MODEL_PATH = str(ROOT / "models" / "Qwen3-32B")


@dataclass
class Skill:
    name: str
    description: str
    focus: str
    head_prior: str
    tail_prior: str
    relation_style: str
    negative_scope: str
    keywords: List[str]
    fewshot: List[Dict[str, str]]


SKILLS: List[Skill] = [
    Skill(
        name="force-organization",
        description="抽取兵力编成、部署位置、主防位置、推进轴线、配属支援、指挥控制与通信火控节点关系。",
        focus="兵力组织、部署与主防位置、推进方向、配属支援、统一指挥、节制转换、节点接续。",
        head_prior="作战单位、敌我旅群、营连分队、指挥所、协调中心、火控节点、通信节点。",
        tail_prior="地点、区域、轴线、被支援单位、被指挥单位、主防地域、接收节点、配属单位。",
        relation_style="关系词应短、稳定，优先表达部署、防御、集结、推进、配属、统一指挥、受控于、接收、转入节制等语义。",
        negative_scope="不要把失败条件、时间窗口、阶段时间、风险限制整句吸收到本 skill。",
        keywords=["下辖", "部署", "集结", "负责", "主防御", "位于", "营", "旅", "分队", "旅群", "统一指挥", "控制", "节制", "火控节点", "通信"],
        fewshot=[
            {
                "text": "第一机步营由西侧盐盘集结区沿峡谷主路北推，负责夺控灰脊桥。",
                "json": '{"head":"第一机步营","relation":"集结于","tail":"西侧盐盘集结区","evidence":"第一机步营由西侧盐盘集结区沿峡谷主路北推","skill":"force-organization"}',
            },
            {
                "text": "本次行动由前沿联合战术群统一指挥，第一机步营受旅前指直接控制。",
                "json": '{"head":"前沿联合战术群","relation":"统一指挥","tail":"本次行动","evidence":"本次行动由前沿联合战术群统一指挥","skill":"force-organization"}',
            },
            {
                "text": "敌主防御位于灰脊桥北坡、下湾桥东岸和折线山口补给站。",
                "json": '{"head":"敌主防御","relation":"位于","tail":"灰脊桥北坡","evidence":"其主防御位于灰脊桥北坡、下湾桥东岸和折线山口补给站","skill":"force-organization"}',
            },
        ],
    ),
    Skill(
        name="operation-constraint",
        description="抽取行动目标、阶段任务、阶段时间、时间窗口、失败条件、先后约束、禁止事项与风险限制。",
        focus="文档级目标、阶段计划、阶段-时间、阶段-任务、成败判据、前提条件、对象受限关系、风险源影响。",
        head_prior="本次行动、本阶段行动、各阶段、具体任务项、桥面、补给车、通信链路、风险源。",
        tail_prior="时间窗口、目标对象、任务项、失败状态、约束短语、被影响对象、限制对象。",
        relation_style="关系词保持开放，但应短而明确，优先使用开始于、结束于、任务、失败于、不得、受限于、遮蔽、覆盖、优先于等结构化写法。",
        negative_scope="不要把纯编成部署、纯指挥链、纯侦察火力节点关系全部吸收到本 skill。",
        keywords=["主要目标", "次要目标", "失败条件", "阶段", "开始于", "结束于", "时间窗口", "前", "后", "不得", "必须", "若", "否则", "超过", "风险", "受限", "优先"],
        fewshot=[
            {
                "text": "第一阶段：渗透与侦察（05:30—09:00），旅侦察分队前出占领西侧观察高地。",
                "json": '{"head":"第一阶段","relation":"开始于","tail":"05:30","evidence":"第一阶段：渗透与侦察（05:30—09:00）","skill":"operation-constraint"}',
            },
            {
                "text": "本次行动失败条件之一是主补给线未被实际阻断。",
                "json": '{"head":"本次行动","relation":"失败于","tail":"主补给线未被实际阻断","evidence":"本次行动失败条件之一是主补给线未被实际阻断","skill":"operation-constraint"}',
            },
            {
                "text": "桥面承载有限，重装备通过顺序必须受控。",
                "json": '{"head":"重装备通过顺序","relation":"受限于","tail":"桥面承载限制","evidence":"桥面承载有限，重装备通过顺序必须受控","skill":"operation-constraint"}',
            },
        ],
    ),
    Skill(
        name="engagement-effects",
        description="抽取敌情展开、侦察发现、火力打击、电子干扰、补给/桥梁/道路等关键节点的作用关系与作战效果。",
        focus="敌编成与部署展开、火力覆盖、干扰压制、侦察回传、校射验证、打击阻断、封桥断补、节点作用对象。",
        head_prior="敌旅群、远程火箭营、干扰节点、侦察平台、火力单元、工兵分队、作战分队、桥梁补给线等关键节点。",
        tail_prior="目标点、阵地、桥梁、补给线、数据链、通信链路、回传链路、保障节点、被打击对象。",
        relation_style="关系词保持开放，但应像覆盖、压制、干扰、发现、回传、校射、切断、封锁、遮断、阻断、保障、摧毁等直接作用。",
        negative_scope="不要把单纯时间窗、失败条件、阶段时间、抽象原则整句吸收到本 skill。",
        keywords=["侦察", "无人机", "火力", "雷达", "干扰", "压制", "回传", "电子", "校射", "火箭", "覆盖", "封锁", "切断", "桥", "补给线"],
        fewshot=[
            {
                "text": "远程火箭位于北侧隐蔽发射区，可覆盖我集结带。",
                "json": '{"head":"远程火箭营","relation":"覆盖","tail":"我集结带","evidence":"远程火箭位于北侧隐蔽发射区，可覆盖我集结带","skill":"engagement-effects"}',
            },
            {
                "text": "电子干扰节点群重点压制卫星中继、无人机回传与旅级数据链。",
                "json": '{"head":"电子干扰节点群","relation":"压制","tail":"无人机回传链路","evidence":"干扰节点重点压制卫星中继、无人机回传与旅级数据链","skill":"engagement-effects"}',
            },
            {
                "text": "工兵分队在两桥南端设置障碍与遥控爆破点，禁止敌抢修车靠近。",
                "json": '{"head":"工兵分队","relation":"封控","tail":"两桥南端","evidence":"工兵分队在两桥南端设置障碍与遥控爆破点","skill":"engagement-effects"}',
            },
        ],
    ),
]


class LocalQwenGenerator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

    def load(self):
        if self.model is not None and self.tokenizer is not None:
            return
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            trust_remote_code=True,
        )

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        self.load()
        assert self.tokenizer is not None and self.model is not None
        messages = [
            {"role": "system", "content": "/no_think 你是一个严格遵循输出格式的军事关系抽取系统。"},
            {"role": "user", "content": "/no_think\n请只输出一个合法 JSON 对象，第一字符必须是 {，最后字符必须是 }。\n" + prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return response


def keyword_scores(text: str) -> Dict[str, int]:
    scores = {}
    for skill in SKILLS:
        scores[skill.name] = sum(text.count(keyword) for keyword in skill.keywords)
    return scores


def build_router_prompt(doc: str, scores: Dict[str, int]) -> str:
    skill_desc = []
    for skill in SKILLS:
        skill_desc.append(
            f"- {skill.name}: {skill.description} 关注重点={skill.focus} 关键词先验分数={scores[skill.name]}"
        )
    return f"""你是一个军事文档技能路由器。请判断下面文档最适合哪些抽取 skills。

可选 skills：
{chr(10).join(skill_desc)}

要求：
1. 必须严格使用中文分析。
2. 只输出 JSON，不要解释，不要 markdown。
3. 输出格式：
{{
  "primary_skill": "skill-name",
  "aux_skills": ["skill-name1", "skill-name2"],
  "reason": "中文简短说明"
}}
4. `aux_skills` 最多两个，可以为空，总技能数最多 3 个。
5. 优先根据文档主要信息密度决定，而不是机械按关键词分数排序。
6. 如果文档同时明显包含兵力/指挥、阶段/约束、侦察火力/敌情作用三类信息，应覆盖这些主要语义面，不要只选一个抽象技能。

文档：
{doc}
"""


def parse_json(content: str) -> Dict:
    text = content.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                recovered = recover_relation_list(candidate)
                if recovered is not None:
                    return recovered
        recovered = recover_relation_list(text)
        if recovered is not None:
            return recovered
        raise


def recover_relation_list(text: str) -> Optional[Dict]:
    key_pos = text.find('"relation_list"')
    if key_pos < 0:
        return None
    array_start = text.find("[", key_pos)
    if array_start < 0:
        return None
    items = []
    depth = 0
    in_string = False
    escape = False
    obj_start = None
    for idx in range(array_start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                obj_start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start is not None:
                    chunk = text[obj_start : idx + 1]
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict):
                        items.append(obj)
                    obj_start = None
        elif ch == "]" and depth == 0:
            break
    if items:
        return {"relation_list": items}
    return None


def generate_text(
    prompt: str,
    backend: str,
    api_client: Optional[OpenAI],
    local_generator: Optional[LocalQwenGenerator],
    model: str,
    max_tokens: int,
) -> str:
    if backend in {"api", "qwen_api"}:
        assert api_client is not None
        request_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0 if max_tokens <= 600 else 0.1,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if backend == "qwen_api":
            request_kwargs["extra_body"] = {"enable_thinking": False}
        response = api_client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content
    assert local_generator is not None
    return local_generator.generate(prompt, max_new_tokens=max_tokens)


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_route_cache(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_route_cache(path: Path, cache: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def route_document(
    backend: str,
    client: Optional[OpenAI],
    local_generator: Optional[LocalQwenGenerator],
    text: str,
    route_cache: Dict,
    model: str,
    max_skills: int = 3,
) -> Dict:
    scores = keyword_scores(text)
    cache_key = doc_hash(text)
    cached = route_cache.get(cache_key)
    valid_skill_names = {skill.name for skill in SKILLS}
    if isinstance(cached, dict):
        ordered = []
        for name in cached.get("selected_skills", []):
            if name in valid_skill_names and name not in ordered:
                ordered.append(name)
        if ordered:
            selected = [skill for skill in SKILLS if skill.name in ordered]
            selected.sort(key=lambda s: ordered.index(s.name))
            return {
                "selected_skills": selected,
                "scores": cached.get("scores", scores),
                "router_reason": cached.get("router_reason", "命中路由缓存。"),
                "cache_hit": True,
            }

    prompt = build_router_prompt(text[:3000], scores)
    try:
        content = generate_text(
            prompt=prompt,
            backend=backend,
            api_client=client,
            local_generator=local_generator,
            model=model,
            max_tokens=160 if backend == "local_qwen3" else 500,
        )
        parsed = parse_json(content)
        ordered = []
        primary = parsed.get("primary_skill")
        if primary in {skill.name for skill in SKILLS}:
            ordered.append(primary)
        for name in parsed.get("aux_skills", []):
            if name in {skill.name for skill in SKILLS} and name not in ordered:
                ordered.append(name)
        if not ordered:
            raise ValueError("router returned empty skills")
        ordered = ordered[:max_skills]
        selected = [skill for skill in SKILLS if skill.name in ordered]
        selected.sort(key=lambda s: ordered.index(s.name))
        result = {
            "selected_skills": selected,
            "scores": scores,
            "router_reason": parsed.get("reason", ""),
            "cache_hit": False,
        }
        route_cache[cache_key] = {
            "selected_skills": [skill.name for skill in selected],
            "scores": scores,
            "router_reason": parsed.get("reason", ""),
        }
        return result
    except Exception:
        ranked = sorted(SKILLS, key=lambda s: scores[s.name], reverse=True)
        selected = [ranked[0]]
        for skill in ranked[1:]:
            if len(selected) >= max_skills:
                break
            if scores[skill.name] >= 2:
                selected.append(skill)
        result = {
            "selected_skills": selected,
            "scores": scores,
            "router_reason": "路由器解析失败，回退到关键词兜底。",
            "cache_hit": False,
        }
        route_cache[cache_key] = {
            "selected_skills": [skill.name for skill in selected],
            "scores": scores,
            "router_reason": "路由器解析失败，回退到关键词兜底。",
        }
        return result


def build_fewshot_block(selected_skills: List[Skill]) -> str:
    lines = []
    for skill in selected_skills:
        lines.append(f"### {skill.name}")
        for example in skill.fewshot:
            lines.append(f"文本片段：{example['text']}")
            lines.append(f"输出示例：{example['json']}")
    return "\n".join(lines)


def build_extraction_prompt(doc: str, selected_skills: List[Skill]) -> str:
    skill_block = []
    for skill in selected_skills:
        skill_block.append(
            f"- {skill.name}: {skill.description}\n"
            f"  关注重点：{skill.focus}\n"
            f"  head 优先类型：{skill.head_prior}\n"
            f"  tail 优先类型：{skill.tail_prior}\n"
            f"  关系词风格：{skill.relation_style}\n"
            f"  不应吸收：{skill.negative_scope}"
        )
    fewshot_block = build_fewshot_block(selected_skills)

    return f"""你是一个军事领域关系抽取系统。

你必须严格使用中文进行抽取，关系名、实体名、证据、skill 字段都必须是中文或原文中的中文短语，不要输出英文解释。

已匹配到以下 skills：
{chr(10).join(skill_block)}

few-shot 示例：
{fewshot_block}

输出要求：
1. 尽量完整抽取文档中有明确证据支持的关系，不要只保留少量“最重要”关系。
2. 输出必须严格使用固定 JSON 结构：
{{
  "relation_list": [
    {{
      "head": "...",
      "relation": "...",
      "tail": "...",
      "evidence": "...",
      "skill": "..."
    }}
  ]
}}
3. `skill` 必须从以下值中选择：{", ".join(skill.name for skill in selected_skills)}。
4. 这是开放域关系抽取，不要把关系词限制成固定清单；但关系词必须简短、稳定、军事语义明确，最好是 2 到 8 个字的动作或作用短语。
5. 优先抽取单位-职责、单位-部署位置、主防位置、指挥控制、阶段任务、阶段时间、失败条件、风险限制、敌火力覆盖、干扰压制、侦察/回传/校射、补给线与桥梁节点作用关系，不要只写任务摘要。
6. 阶段信息必须尽量拆成两类关系：`阶段-开始于/结束于-时间` 与 `阶段-任务-行动项`。不要把阶段整段说明直接塞进 tail。
7. 失败条件、禁止事项、先后约束、对象受限关系必须单列，不能因为它们不像典型三元组就省略。
8. 敌情必须尽量展开：敌编成、主防位置、远程火箭覆盖对象、干扰节点压制对象、补给/桥梁/道路关键节点作用关系应分别抽取。
9. 风险和约束优先写成 `风险源-影响对象`、`对象-受限于-约束`、`任务项-失败于-条件` 这类结构化关系。
10. 如果一句话里包含多个并列目标、多个节点、多个对象，能合理拆开时尽量拆成多条关系；优先拆成实体关系，不要把整句目标塞进一个 tail。
11. 不要输出以下不合理内容：
   - 只有摘要意义、没有明确对象的泛关系，如“我方-实施-联合突击”
   - 以“主要目标/次要目标/失败条件/优先原则”作为 head 的关系
   - 把整句条件、整句任务、整句原则原样作为 head 或 tail
   - 关系词过长、像自然语言短句，而不是稳定关系短语
12. `head` 和 `tail` 应尽量写成合理的实体、地点、设施、线路、任务项或条件项，不要整段照抄长句。
13. 尽量少用“本次行动”这类泛主语；只有在确实表达文档级目标或约束条件，且不存在更具体执行主体时，才使用“本次行动”或“本阶段行动”。
14. 如果句子同时含有“行动概述”和“具体作用对象”，优先保留更接近实体关系的写法。例如优先写“远程火箭营-覆盖-我集结带”，而不是“敌方-具备能力-覆盖我集结带”。
15. 不限制输出条数。不要过度过滤；只要有直接证据且语义基本成立，就可以保留。重复关系再去掉即可。
16. 对于以下容易漏掉的关系类型，要主动检查并尽量补全：
   - 通信主链路、备份链路、侦察链路、回传链路、火控节点、目标指示
   - 高价值目标的使用限制、禁止事项、先决条件、对象受限关系
   - 敌方远程火箭、干扰节点、桥梁、道路、补给线、通路的作用对象
   - 各阶段中的具体动作项，而不只是阶段标题和时间
17. 如果同一句中存在多个并列对象，优先拆成多条关系。例如“压制卫星中继、无人机回传与旅级数据链”应尽量拆开。
18. 在不牺牲准确性的前提下，优先保证召回；后续会做去重整理，因此不要因为担心重复而少抽。
19. 只返回 JSON，不要 markdown，不要解释。

文档：
{doc}
"""


def build_refinement_prompt(
    doc: str,
    selected_skills: List[Skill],
    relation_list: List[Dict[str, str]],
) -> str:
    skill_block = []
    for skill in selected_skills:
        skill_block.append(
            f"- {skill.name}: 关注重点={skill.focus}；关系词风格={skill.relation_style}；不应吸收={skill.negative_scope}"
        )
    relation_json = json.dumps({"relation_list": relation_list}, ensure_ascii=False, indent=2)
    return f"""你是一个军事关系抽取结果整理器。下面给你文档和一批候选关系，请你只做整理，不要凭空新增文档里没有证据的新事实。

已选 skills：
{chr(10).join(skill_block)}

整理要求：
1. 保留开放域关系词，不要强行改成固定标签。
2. 删除明显重复、空泛或证据不足的关系，但默认应保留能在原文中直接定位证据的候选关系，不要过度删减。
3. 如果 `head` 或 `tail` 过长且像整句任务描述，请压缩成更短的实体或任务对象短语。
4. 如果一句候选关系其实包含多个并列对象，且文档证据充分，可以拆成多条。
5. 如果某条关系更适合另一个已选 skill，可以改 `skill`。
6. 优先保留实体之间、实体与设施/线路之间、任务项与约束条件之间的直接关系。
7. 去掉摘要式关系，如“我方-实施-联合突击”“本次行动-主要目标-压制某目标”这类可被更具体关系替代的写法。
8. 保留阶段时间、失败条件、风险限制、敌火力/干扰作用对象等结构化信息，不要在整理时误删。
9. 如果某条关系本身是有价值的具体关系，但表达还不够规范，优先改写，不要直接删除。
10. 如果原候选中已经有较具体的对象级关系，不要退化成摘要式关系。
11. 只输出 JSON，不要解释。

文档：
{doc}

候选关系：
{relation_json}
"""


def extract_chunk(
    backend: str,
    client: Optional[OpenAI],
    local_generator: Optional[LocalQwenGenerator],
    model: str,
    chunk_index: int,
    chunk_text: str,
    selected_skills: List[Skill],
) -> Dict:
    prompt = build_extraction_prompt(chunk_text, selected_skills)
    max_tokens = 800 if backend == "local_qwen3" else 4200
    if backend == "qwen_api":
        max_tokens = 2200
    content = generate_text(
        prompt=prompt,
        backend=backend,
        api_client=client,
        local_generator=local_generator,
        model=model,
        max_tokens=max_tokens,
    )
    parsed_chunk = parse_json(content)
    return {
        "chunk_index": chunk_index,
        "chunk_token_count": approx_token_count(chunk_text),
        "prediction": parsed_chunk,
    }


def load_samples(path: Path, limit: int) -> List[Dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if len(samples) >= limit:
                break
            obj = json.loads(line)
            samples.append(
                {
                    "sample_index": idx,
                    "sample_id": obj.get("sample_id"),
                    "input_meta": obj.get("input", {}),
                    "doc_text": obj.get("output", ""),
                }
            )
    return samples


def approx_token_count(text: str) -> int:
    # Approximate mixed Chinese/English token count for routing chunking decisions.
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_tokens = re.findall(r"[A-Za-z0-9_+\-]+", text)
    punctuation = re.findall(r"[，。！？；：,.!?;:（）()【】\[\]“”\"'、】【\-]", text)
    return cjk_chars + len(latin_tokens) + max(1, len(punctuation) // 4)


def split_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?])\s+|(?<=[。！？!?])|(?<=\.)\s+\n*|\n+", text) if s.strip()]
    if not sentences:
        return [text]
    return sentences


def chunk_document(text: str, token_budget: int) -> List[str]:
    sentences = split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = approx_token_count(sentence)
        if current and current_tokens + sent_tokens > token_budget:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sent_tokens
        else:
            current.append(sentence)
            current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def normalize_relation_text(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def looks_like_clause(text: str) -> bool:
    if len(text) <= 20:
        return False
    markers = ["并", "且", "以便", "用于", "确保", "随后", "同时", "若", "则", "必须", "任务", "行动"]
    return any(marker in text for marker in markers)


def looks_like_bad_tail(text: str) -> bool:
    if len(text) <= 28:
        return False
    markers = ["；", "。", "，且", "，并", "否则", "以防", "优先", "原则", "行动", "任务通报"]
    return any(marker in text for marker in markers)


def split_compound_tail(text: str) -> List[str]:
    if not text:
        return []
    candidate = text.strip(" ，。；：,.;")
    if not candidate:
        return []
    parts = [
        item.strip(" ，。；：,.;")
        for item in re.split(r"[、]|以及|及|与|和", candidate)
    ]
    parts = [item for item in parts if item]
    if len(parts) < 2:
        return []
    if any(len(item) > 24 for item in parts):
        return []
    return parts


def expand_relation_item(rel: Dict[str, str]) -> List[Dict[str, str]]:
    relation = rel.get("relation", "")
    tail = rel.get("tail", "")
    evidence = rel.get("evidence", "")
    skill = rel.get("skill", "")
    splittable_relations = {
        "位于",
        "部署于",
        "集结于",
        "压制",
        "覆盖",
        "封锁",
        "切断",
        "依托",
        "连接",
        "保障",
    }
    if relation not in splittable_relations:
        return [rel]
    if skill not in {"force-organization", "engagement-effects"}:
        return [rel]
    parts = split_compound_tail(tail)
    if not parts:
        return [rel]
    matched_parts = [part for part in parts if part in evidence]
    if len(matched_parts) < 2:
        return [rel]
    expanded = []
    for part in matched_parts:
        item = rel.copy()
        item["tail"] = part
        expanded.append(item)
    return expanded if len(expanded) >= 2 else [rel]


def normalize_relation_phrase(head: str, relation: str, tail: str) -> str:
    relation = relation.strip()
    simple_map = {
        "受控制": "受控于",
        "接受": "接收",
        "位于于": "位于",
    }
    relation = simple_map.get(relation, relation)
    if relation == "部署" and any(token in tail for token in ["区", "桥", "岸", "高地", "坡", "站", "线", "口", "带", "地域"]):
        return "部署于"
    if relation == "负责" and len(tail) <= 10:
        return "任务"
    return relation


def select_evidence(head: str, relation: str, tail: str, evidence: str) -> str:
    candidates = [evidence.strip(), f"{head}{relation}{tail}".strip()]
    candidates = [item for item in candidates if item]
    if not candidates:
        return ""
    return min(candidates, key=len)


def sanitize_relation_item(rel: Dict) -> Optional[Dict[str, str]]:
    head = rel.get("head", "").strip(" ，。；：,.;")
    tail = rel.get("tail", "").strip(" ，。；：,.;")
    relation = normalize_relation_phrase(head, rel.get("relation", "").strip(" ，。；：,.;"), tail)
    evidence = rel.get("evidence", "").strip()
    skill = rel.get("skill", "").strip()
    if not head or not relation or not tail or not skill:
        return None
    if len(relation) > 12:
        return None
    if head in {"主要目标", "次要目标", "失败条件", "时间约束", "优先原则"}:
        return None
    if relation in {"为", "由", "是"}:
        return None
    if looks_like_clause(head) and head not in {"本次行动", "本阶段行动"}:
        return None
    if looks_like_bad_tail(tail):
        return None
    if head in {"我方", "敌方"} and relation in {"实施", "拟实施", "目标为", "任务", "计划"}:
        return None
    if relation in {"主要目标", "次要目标", "失败条件"}:
        return None
    if skill not in {"force-organization", "operation-constraint", "engagement-effects"}:
        return None
    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "evidence": select_evidence(head, relation, tail, evidence),
        "skill": skill,
    }


def merge_chunk_relations(chunk_results: List[Dict]) -> Dict:
    merged = []
    seen = {}
    for item in chunk_results:
        for rel in item.get("relation_list", []):
            sanitized = sanitize_relation_item(rel)
            if not sanitized:
                continue
            for expanded in expand_relation_item(sanitized):
                head = normalize_relation_text(expanded.get("head", ""))
                relation = normalize_relation_text(expanded.get("relation", ""))
                tail = normalize_relation_text(expanded.get("tail", ""))
                skill = normalize_relation_text(expanded.get("skill", ""))
                if not head or not relation or not tail:
                    continue
                key = (head, relation, tail, skill)
                evidence = expanded.get("evidence", "").strip()
                if key not in seen:
                    stored = expanded
                    seen[key] = stored
                    merged.append(stored)
                else:
                    existing = seen[key]
                    if evidence and (not existing["evidence"] or len(evidence) < len(existing["evidence"])):
                        existing["evidence"] = evidence
    return {"relation_list": merged}


def normalize_prediction(prediction: Dict) -> Dict:
    return merge_chunk_relations([prediction])


def refine_prediction(
    backend: str,
    client: Optional[OpenAI],
    local_generator: Optional[LocalQwenGenerator],
    model: str,
    doc: str,
    selected_skills: List[Skill],
    prediction: Dict,
) -> Dict:
    relation_list = prediction.get("relation_list", [])
    if not relation_list:
        return prediction
    prompt = build_refinement_prompt(doc, selected_skills, relation_list)
    try:
        max_tokens = 1000 if backend == "local_qwen3" else 3000
        if backend == "qwen_api":
            max_tokens = 2200
        content = generate_text(
            prompt=prompt,
            backend=backend,
            api_client=client,
            local_generator=local_generator,
            model=model,
            max_tokens=max_tokens,
        )
        refined = parse_json(content)
        return merge_chunk_relations([prediction, refined])
    except Exception:
        return normalize_prediction(prediction)


def main():
    parser = argparse.ArgumentParser(description="Run military skill-router extraction trial.")
    parser.add_argument("--input", default=str(ROOT / "data" / "fight_data.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "examples" / "results" / "fight_data_skill_router_trial.json"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--chunk-trigger", type=int, default=DEFAULT_CHUNK_TRIGGER)
    parser.add_argument("--chunk-budget", type=int, default=DEFAULT_CHUNK_BUDGET)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--backend", choices=["api", "qwen_api", "local_qwen3"], default="api")
    parser.add_argument("--local-model-path", default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument(
        "--route-cache",
        default=str(ROOT / "examples" / "results" / "fight_data_route_cache.json"),
    )
    args = parser.parse_args()
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if args.backend in {"api", "qwen_api"} and not api_key:
        parser.error("--api-key is required when --backend api/qwen_api")

    client = None
    local_generator = None
    if args.backend == "api":
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            http_client=httpx.Client(trust_env=False, timeout=120.0),
        )
    elif args.backend == "qwen_api":
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=httpx.Client(trust_env=False, timeout=120.0),
        )
    else:
        local_generator = LocalQwenGenerator(args.local_model_path)
    route_cache_path = Path(args.route_cache)
    route_cache = load_route_cache(route_cache_path)

    samples = load_samples(Path(args.input), args.limit)
    results = []

    for sample in samples:
        doc_text = sample["doc_text"]
        routing = route_document(
            backend=args.backend,
            client=client,
            local_generator=local_generator,
            text=doc_text,
            route_cache=route_cache,
            model=args.model,
        )
        selected_skills = routing["selected_skills"]
        doc_tokens = approx_token_count(doc_text)
        used_chunking = doc_tokens > args.chunk_trigger
        if used_chunking:
            chunks = chunk_document(doc_text, args.chunk_budget)
            chunk_predictions = []
            max_workers = max(1, min(args.max_workers, len(chunks)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        extract_chunk,
                        args.backend,
                        client,
                        local_generator,
                        args.model,
                        chunk_index,
                        chunk_text,
                        selected_skills,
                    )
                    for chunk_index, chunk_text in enumerate(chunks)
                ]
                for future in as_completed(futures):
                    chunk_predictions.append(future.result())
            chunk_predictions.sort(key=lambda x: x["chunk_index"])
            parsed = merge_chunk_relations([item["prediction"] for item in chunk_predictions])
        else:
            chunks = [doc_text]
            chunk_predictions = []
            single_result = extract_chunk(
                args.backend,
                client,
                local_generator,
                args.model,
                0,
                doc_text,
                selected_skills,
            )
            parsed = normalize_prediction(single_result["prediction"])
        parsed = refine_prediction(
            backend=args.backend,
            client=client,
            local_generator=local_generator,
            model=args.model,
            doc=doc_text,
            selected_skills=selected_skills,
            prediction=parsed,
        )
        results.append(
            {
                "sample_index": sample["sample_index"],
                "sample_id": sample["sample_id"],
                "input_meta": sample["input_meta"],
                "preprocess": {
                    "doc_token_count": doc_tokens,
                    "used_chunking": used_chunking,
                    "chunk_count": len(chunks),
                    "chunk_trigger": args.chunk_trigger,
                    "chunk_budget": args.chunk_budget,
                },
                "routing": {
                    "selected_skills": [skill.name for skill in selected_skills],
                    "scores": routing["scores"],
                    "router_reason": routing["router_reason"],
                    "cache_hit": routing["cache_hit"],
                },
                "chunk_predictions": chunk_predictions,
                "prediction": parsed,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    save_route_cache(route_cache_path, route_cache)

    print(output_path)
    for item in results:
        rels = item["prediction"].get("relation_list", [])
        print(
            f"sample={item['sample_id']} skills={','.join(item['routing']['selected_skills'])} relation_count={len(rels)}"
        )


if __name__ == "__main__":
    main()
