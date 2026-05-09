import json
from pathlib import Path
from typing import List

from skill4re.models import Skill


def load_skills(skills_dir: Path) -> List[Skill]:
    skill_files = sorted(skills_dir.glob("*.json"))
    if not skill_files:
        raise FileNotFoundError(f"No skill json files found in {skills_dir}")
    skills = []
    for path in skill_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        skills.append(Skill.from_dict(payload))
    return skills
