import sys
from pathlib import Path


SKILL4RE_ROOT = Path(__file__).resolve().parents[2] / "skill4re"
if str(SKILL4RE_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL4RE_ROOT))

__all__ = ["create_app"]


def create_app():
    from .api import create_app as _create_app

    return _create_app()
