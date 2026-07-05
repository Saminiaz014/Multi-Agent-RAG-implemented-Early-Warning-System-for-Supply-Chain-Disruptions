"""Supply Chain DSS — dashboard entry point (redesigned, two pages).

Run with::

    streamlit run src/dashboard/app.py

Two pages, two audiences (see the module docstrings for detail):

* **Decision View** (default) — no-scroll, plain-language operational view for
  a supply-chain manager: status words, a recommended action, a 3-D chokepoint
  map, and a natural-language risk explanation.  No raw scores.
* **Analysis View** — scrollable research page for the thesis author: all 8
  Phase 9a evaluation metrics, raw signals, SHAP values, hand-tuned vs
  optimized comparisons, and per-chart JPEG export.

Structure: true Streamlit multipage via ``st.navigation`` over thin page
shims in ``pages/``; all logic lives in importable modules
(``core.py`` / ``decision_view.py`` / ``analysis_view.py``) so pytest can
exercise it without a Streamlit runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Re-exports — keep the Phase 9b test surface (and any external callers) stable.
from src.dashboard.core import (  # noqa: E402,F401
    ACTIONS,
    AGENT_COLORS,
    ALL_AGENTS,
    AVAILABLE_REGIONS,
    GLOBE_FALLBACK_MESSAGE,
    RANGE_PRESETS,
    RISK_COLORS,
    SCENARIOS,
    STATUS_COLORS,
    build_globe_html,
    classify_level,
    decide_action,
    fig_to_jpeg,
    get_monitoring_points,
    get_news,
    get_routes,
    get_vessels,
    predict_action,
    preset_to_range,
    resolve_cesium_token,
    select_route,
    status_word,
)


def main() -> None:
    st.set_page_config(
        page_title="Supply Chain DSS — Strait of Hormuz",
        page_icon="🛳",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    pages = st.navigation([
        st.Page("pages/1_Decision_View.py", title="Decision View", icon="🧭", default=True),
        st.Page("pages/2_Analysis_View.py", title="Analysis View", icon="📊"),
    ])
    pages.run()


if __name__ == "__main__":
    main()
