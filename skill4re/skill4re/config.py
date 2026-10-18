from pathlib import Path
from typing import Dict


SKILL4RE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SKILL4RE_ROOT.parent
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "fight_data.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "skill4re" / "results" / "fight_data_skill4re.json"
DEFAULT_ROUTE_CACHE_PATH = REPO_ROOT / "skill4re" / "results" / "fight_data_route_cache.json"
DEFAULT_SKILLS_DIR = SKILL4RE_ROOT / "skills"
DEFAULT_CHUNK_BUDGET = 900
DEFAULT_CHUNK_TRIGGER = 1200
DEFAULT_MAX_WORKERS = 4
DEFAULT_LOCAL_MODEL_PATH = str(REPO_ROOT / "models" / "Qwen3-32B")

# 路由、共指消解、抽取、校对、汇总、定向校对各阶段的输出上限，集中在本表维护。
# ANSWER_MAX_TOKENS 是答案本身需要预留的长度；自部署端点（vllm）开启思考时隐藏的思考
# token 会与答案共用同一个 max_tokens 配额，因此这类后端在答案上限之外再加
# THINKING_MAX_TOKENS 的思考预留。百炼兼容端点（qwen_api）的思考长度由 reasoning_effort
# 单独控制，不再叠加预留。本机量化推理（local_qwen3）的思考由 /no_think 关闭，
# 上限按本机速度与答案规模单独设定。
ANSWER_MAX_TOKENS: Dict[str, int] = {
    "entity_recognition": 16384,
    "entity_proofreading": 16384,
    "entity_merge": 8192,
    "routing": 500,
    "coref": 3600,
    "extraction": 16384,
    "proofreading": 16384,
    "summarize": 16384,
    "targeted_proofread": 8192,
}
THINKING_MAX_TOKENS = 8192
LOCAL_MAX_TOKENS: Dict[str, int] = {
    "entity_recognition": 2400,
    "entity_proofreading": 2400,
    "entity_merge": 1600,
    "routing": 160,
    "coref": 1600,
    "extraction": 2400,
    "proofreading": 2400,
    "summarize": 3200,
    "targeted_proofread": 1600,
}
QUOTA_SHARING_BACKENDS = {"vllm"}


def max_tokens_for(task_type: str, backend: str) -> int:
    """取某个阶段的输出上限；未知阶段直接报错。"""
    if task_type not in ANSWER_MAX_TOKENS:
        raise ValueError(f"未知的请求阶段：{task_type}")
    if backend == "local_qwen3":
        return LOCAL_MAX_TOKENS[task_type]
    limit = ANSWER_MAX_TOKENS[task_type]
    if backend in QUOTA_SHARING_BACKENDS:
        limit += THINKING_MAX_TOKENS
    return limit
