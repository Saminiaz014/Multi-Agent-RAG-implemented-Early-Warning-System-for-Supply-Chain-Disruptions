"""Page shim — Analysis view. All logic lives in src/dashboard/analysis_view.py."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.analysis_view import render  # noqa: E402

render()
