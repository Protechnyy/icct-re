from pathlib import Path


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
