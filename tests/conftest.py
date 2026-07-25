"""Ensures the project root is importable as `utils`, `nlp`, `market`, etc. even if
pyproject.toml's pythonpath setting isn't honored by the runner (e.g. an IDE test
runner that ignores pyproject.toml).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
