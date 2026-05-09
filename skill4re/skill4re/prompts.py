import json
import re
from typing import Dict, List

from skill4re.models import Skill


BASE_EXTRACTION_RULES = [
    "只抽取原文有明确证据支持的关系，优先保证召回，后处理会去重。",
    "输出固定 JSON：{\"relation_list\":[{\"head\":\"...\",\"relation\":\"...\",\"tail\":\"...\",\"evidence\":\"...\",\"skill\":\"...\"}]}。",
    "字段必须使用中文或原文中文短语；relation 要短、稳定、语义明确，通常 2 到 8 个字。",
    "head/tail 尽量是实体、组织、人物、设施、事件、概念或条件项，不要整句照抄原文。",
    "一句话含多个并列对象且证据充分时，拆成多条实体关系。",
    "不要输出摘要式泛关系，如 `X-实施-Y` `X-涉及-Y` 这类无信息量的写法。",
    "evidence 必须是原文中可定位的片段，不要改写或概括。",
    "关系方向要自然：主动方做 head，被动方或目标做 tail。",
    "同一实体的不同表述统一为最完整的名称（如 `该营` 统一为 `第3机械化步兵营`）。",
    "只返回 JSON，不要 markdown，不要解释。",
    "【重要】关系谓词要精准，避免使用过泛的词：不要用 '必须'、'关键'、'威胁'、'涉及'、'包含' 等，改用更具体的动词如 '部署于'、'装备'、'依赖'、'导致'、'保护' 等。",
    "【重要】因果关系方向要正确：原因/条件做 head，结果/影响做 tail。例如：'基地车损毁-导致-行动中止' 而不是 '基地车-损毁于-行动失败'。",
    "【重要】基础元信息必须抽取：文件等级、作战区域、行动代号、行动时间、参战单位等。",
]

SKILL_EXTRACTION_RULES = {
    "military": [
        "兵力编成：重点抽取单位-职责、单位-部署/集结位置、主防位置、推进轴线、配属支援。",
        "指挥控制：指挥关系、节制转换、通信/火控节点接续要单列。",
        "火力与对抗：火力覆盖/打击对象、干扰压制对象、侦察回传链路要分别抽取。",
        "阶段与约束：阶段信息拆成 `阶段-开始于/结束于-时间`，失败条件、禁止事项、先后约束必须单列。",
        "关键节点：桥梁、道路、补给线、通路的封锁、切断、保障、阻断作用要单列。",
        "【新增】基础元信息：文件等级、作战区域、行动代号、行动时间、参战单位等必须抽取。",
        "【新增】建设与部署：发电厂、兵营、矿石精炼厂、战车工厂、爱国者导弹、机枪碉堡等建设关系要抽取。",
        "【新增】敌军装备：苏军装备（犀牛坦克、天启坦克、V3火箭发射车、磁暴步兵）和自动防御系统（磁暴线圈、防空炮、哨戒炮、机枪碉堡）要完整抽取。",
        "【新增】弱点关系：敌方依赖电力、依赖心灵信标等弱点关系必须抽取。格式：'敌方-依赖-电力'、'磁暴线圈-失效于-电厂被摧毁'。",
        "【新增】失败条件：基地车损毁、铁幕装置全面启动、敌方主力突破防线等失败条件必须抽取。格式：'基地车损毁-导致-任务失败'。",
    ],
    "finance": [
        "公司关系：股权、控股、参股、子公司、母公司关系要精确抽取。",
        "财务数据：营收、利润、资产等关键指标与对应主体/时间的关系要抽取。",
        "高管与治理：董事、高管的任职、兼任、变更关系要单列。",
        "投资与并购：投资标的、收购对象、交易金额、对赌条款要分别抽取。",
        "风险与处罚：违规、处罚、诉讼、仲裁等负面事件与主体的关系要单列。",
    ],
    "medical": [
        "疾病与症状：疾病-典型症状、疾病-并发症、症状-鉴别诊断关系要分别抽取。",
        "诊疗关系：检查-适应症、手术-适应症、药物-适应症/禁忌症要精确抽取。",
        "用药关系：药物-剂量、药物-不良反应、药物-相互作用要单列。",
        "预后因素：影响预后的因素、风险分层指标要与疾病/治疗方案关联。",
        "医学证据：临床试验-结论、指南-推荐级别、文献-证据等级要抽取。",
    ],
    "legal": [
        "案件结构：原告-被告、上诉人-被上诉人、申请人-被申请人关系要明确。",
        "法律适用：法条-适用事实、罪名-构成要件、裁判依据要精确抽取。",
        "事实认定：证据-证明事实、行为-定性、因果关系链要分别抽取。",
        "裁判结果：判决-主文、责任-承担方式、赔偿-金额关系要单列。",
        "程序关系：审理程序、管辖权、时效、上诉/再审关系要抽取。",
    ],
    "technology": [
        "系统架构：系统-子系统、模块-功能、组件-依赖关系要精确抽取。",
        "接口与协议：API-调用方、协议-端口、数据流-方向要分别抽取。",
        "故障分析：故障-原因、故障-影响范围、告警-故障关系要单列。",
        "运维操作：操作-前置条件、变更-影响范围、回滚-触发条件要抽取。",
        "性能指标：指标-阈值、指标-监控对象、SLA-服务关系要抽取。",
    ],
}


def build_router_prompt(doc: str, scores: Dict[str, float], skills: List[Skill]) -> str:
    skill_desc = []
    for skill in skills:
        skill_desc.append(
            f"- {skill.name}: {skill.description} 关注重点={skill.focus} 关键词加权分数={scores[skill.name]:g}"
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


def fewshot_relevance_score(doc: str, skill: Skill, example_text: str, example_json: str) -> int:
    score = 0
    example_blob = f"{example_text}\n{example_json}"
    for keyword in skill.keywords:
        if not keyword or keyword not in doc:
            continue
        keyword_hits = doc.count(keyword)
        score += keyword_hits
        if keyword in example_blob:
            score += keyword_hits * 3

    doc_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_+\-]{2,}", doc))
    example_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_+\-]{2,}", example_blob))
    score += len(doc_terms & example_terms)
    return score


def build_fewshot_block(selected_skills: List[Skill], doc: str, max_examples: int = 2) -> str:
    """构建 few-shot 示例块。

    优先选择文档级示例（多条关系），按与当前文档的关键词重叠度排序。
    """
    ranked_examples = []
    for skill in selected_skills:
        for example_index, example in enumerate(skill.fewshot):
            ranked_examples.append(
                (
                    fewshot_relevance_score(doc, skill, example.text, example.json),
                    1 if example.is_document_level else 0,  # 文档级示例优先
                    -example_index,
                    skill.name,
                    example,
                )
            )

    if not ranked_examples:
        return "无"

    ranked_examples.sort(reverse=True)
    selected = [item for item in ranked_examples if item[0] > 0][:max_examples]
    if not selected:
        selected = ranked_examples[:1]

    lines = []
    for _, _, _, skill_name, example in selected:
        is_doc_level = example.is_document_level
        lines.append(f"### {skill_name}")
        lines.append(f"{'文档片段' if is_doc_level else '文本片段'}：{example.text}")
        lines.append(f"输出示例：{example.json}")
    return "\n".join(lines)


def build_extraction_rules(selected_skills: List[Skill]) -> str:
    rules = list(BASE_EXTRACTION_RULES)
    for skill in selected_skills:
        # 优先从 skill 对象读取规则，如果没有则从 SKILL_EXTRACTION_RULES 读取
        if skill.extraction_rules:
            rules.extend(skill.extraction_rules)
        else:
            rules.extend(SKILL_EXTRACTION_RULES.get(skill.name, []))
    return "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))


def build_extraction_prompt(doc: str, selected_skills: List[Skill]) -> str:
    skill_block = []
    for skill in selected_skills:
        skill_block.append(
            f"- {skill.name}: 关注={skill.focus}；head={skill.head_prior}；"
            f"tail={skill.tail_prior}；关系={skill.relation_style}；边界={skill.negative_scope}"
        )
    fewshot_block = build_fewshot_block(selected_skills, doc)
    extraction_rules = build_extraction_rules(selected_skills)
    return f"""你是一个军事领域关系抽取系统。

你必须严格使用中文进行抽取，关系名、实体名、证据、skill 字段都必须是中文或原文中的中文短语，不要输出英文解释。

已匹配到以下 skills：
{chr(10).join(skill_block)}

few-shot 示例：
{fewshot_block}

输出要求：
`skill` 必须从以下值中选择：{", ".join(skill.name for skill in selected_skills)}。
{extraction_rules}

文档：
{doc}
"""


def build_proofreading_prompt(
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
    return f"""你是一个军事关系抽取结果校对器。下面给你完整文档和一批候选关系，请在一次校对中同时完成整理、纠偏和少量补漏。

已选 skills：
{chr(10).join(skill_block)}

校对要求：
1. 默认保留原有候选关系。只有明显重复、字段缺失、证据不支持、关系词明显错误、完全摘要化且已被更具体关系覆盖时，才能删除。
2. 不要把大量候选压缩成少量"最安全"的关系。候选能在原文中直接定位证据时，应优先保留或改写。
3. 保留开放域关系词，不要强行改成固定标签；relation 要短、稳定、军事语义明确。
4. 如果 `head` 或 `tail` 过长且像整句任务描述，请压缩成更短的实体、设施、线路、任务项或条件项。
5. 如果一句候选关系其实包含多个并列对象，且文档证据充分，可以拆成多条。
6. 如果某条关系更适合另一个已选 skill，可以改 `skill`；`skill` 只能从以下值中选择：{", ".join(skill.name for skill in selected_skills)}。
7. 可以补充少量原文中直接有证据、但候选遗漏的重要关系，重点检查：
   - 阶段开始/结束时间
   - 阶段任务
   - 失败条件
   - 对象受限关系
   - 干扰压制对象
   - 火力覆盖/打击对象
   - 指挥控制链
   - 桥梁/道路/补给线节点作用
8. 优先保留实体之间、实体与设施/线路之间、任务项与约束条件之间的直接关系。
9. 去掉摘要式关系，如"我方-实施-联合突击""本次行动-主要目标-压制某目标"这类可被更具体关系替代的写法。
10. 保留阶段时间、失败条件、风险限制、敌火力/干扰作用对象等结构化信息，不要在校对时误删。
11. 输出仍必须严格为：
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
12. 只输出 JSON，不要解释，不要 markdown。

文档：
{doc}

候选关系：
{relation_json}
"""


def build_summarize_prompt(
    doc: str,
    selected_skills: List[Skill],
    chunk_results: List[Dict],
) -> str:
    """构建 chunk 结果语义合并 prompt。

    与 proofreading 不同：这里输入是多个 chunk 的独立结果，需要做跨 chunk 的
    实体归一化和关系去重，而不是对单个结果做校对。
    """
    skill_names = ", ".join(skill.name for skill in selected_skills)
    # 把每个 chunk 的结果标上 chunk_index
    results_with_index = []
    for item in chunk_results:
        idx = item.get("chunk_index", "?")
        rels = item.get("relation_list", []) if "relation_list" in item else item
        results_with_index.append({"chunk_index": idx, "relation_list": rels})
    results_json = json.dumps(results_with_index, ensure_ascii=False, indent=2)
    return f"""你是一个军事关系抽取结果合并器。下面给你同一文档不同片段的抽取结果，请做跨片段的语义合并。

合并要求：
1. 同一实体的不同表述统一为最完整的名称（如"第3营"统一为"第3机械化步兵营"）。
2. 重复关系只保留证据最充分、表述最完整的一条。
3. 同一实体在不同片段中的不同关系都要保留（如片段A说"第3营部署于A高地"，片段B说"第3营负责主攻"，应保留两条）。
4. 不要新增文档中没有证据的关系。
5. 如果某条关系的 evidence 在多个片段中都出现，保留最短的那条。
6. skill 只能从以下值中选择：{skill_names}。
7. 输出格式：
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
8. 只输出 JSON，不要解释，不要 markdown。

文档：
{doc}

各片段抽取结果：
{results_json}
"""


def build_targeted_proofreading_prompt(
    doc: str,
    selected_skills: List[Skill],
    low_confidence_relations: List[Dict[str, str]],
) -> str:
    """构建针对低置信度关系的反思 prompt。

    与通用 proofreading 不同：这里只发低置信度的关系，让 LLM 逐条审查，
    而不是对全部结果做大规模改写。
    """
    skill_names = ", ".join(skill.name for skill in selected_skills)
    relation_json = json.dumps({"relation_list": low_confidence_relations}, ensure_ascii=False, indent=2)
    return f"""你是一个军事关系抽取结果反思器。以下是从文档中抽取的低置信度关系，请逐条审查并修正。

审查要求：
1. 对每条关系，检查 head、relation、tail 是否在文档中有明确证据支持。
2. 如果证据不足或关系明显错误，删除该条。
3. 如果实体名不准确（如用了简称而非全称），修正为文档中的完整表述。
4. 如果关系词不准确或过长，修正为更精准的短语。
5. 如果某条实际是正确的，保留它。
6. 不要新增文档中没有证据的关系。
7. skill 只能从以下值中选择：{skill_names}。
8. 输出格式：
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
9. 只输出 JSON，不要解释，不要 markdown。

文档：
{doc}

低置信度关系：
{relation_json}
"""
