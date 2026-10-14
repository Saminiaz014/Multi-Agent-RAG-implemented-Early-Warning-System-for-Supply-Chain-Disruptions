"""One style for every Chapter 4 figure.

* :data:`AGENT_COLORS` — Okabe–Ito hues in a fixed order, so an agent keeps its
  colour in every figure regardless of which agents a panel shows. The order
  was checked with the dataviz palette validator against a white surface: all
  adjacent pairs pass the CVD (deuteranopia ΔE >= 11) and normal-vision
  floors. Orange and pink fall below 3:1 contrast on white, so figures
  using them carry direct labels. The dashboard keeps its own colours
  (``src/dashboard/core.py::AGENT_COLORS``); this map is for thesis figures only.
* :func:`apply` — one rcParams set (fonts, sizes, recessive grid).
* :func:`add_provenance` — the footer every figure carries: data origin, the
  aggregation path explained, and a SYNTHETIC / SIMULATED banner when relevant.
* :func:`save` — writes PNG (300 dpi), PDF and ``.caption.txt`` side by side.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "thesis" / "figures" / "ch4"

#: Agent -> colour, Okabe–Ito, in the validated order.
AGENT_COLORS: dict[str, str] = {
    "shipping": "#0072B2",
    "market": "#E69F00",
    "geopolitical": "#009E73",
    "natural_disaster": "#D55E00",
    "news_sentiment": "#CC79A7",
}
AGENT_ORDER: tuple[str, ...] = tuple(AGENT_COLORS)
AGENT_LABELS: dict[str, str] = {
    "shipping": "Shipping",
    "market": "Market",
    "geopolitical": "Geopolitical",
    "natural_disaster": "Natural disaster",
    "news_sentiment": "News",
}
REGION_LABELS: dict[str, str] = {
    "hormuz": "Strait of Hormuz",
    "bab_el_mandeb": "Bab el-Mandeb",
    "panama": "Panama Canal",
    "malacca": "Strait of Malacca",
}

INK = "#1a1a1a"
INK_MUTED = "#5f5f5f"
GRID = "#e3e3e3"

#: Banners required on figures whose data is not real evaluation data.
_BANNERS = {"synthetic": "SYNTHETIC", "simulated": "SIMULATED OPERATOR"}


def apply() -> None:
    """Set the shared rcParams. Idempotent."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def add_provenance(fig, data_origin: str, path: str, extra: str = "") -> None:
    """Footer naming the data and the explained aggregation path.

    Args:
        fig: The figure.
        data_origin: ``real_eval``, ``synthetic_seed44``, ``simulated``, ...
        path: The aggregation path explained (``"D"`` or ``"C"``), or ``"-"``.
        extra: Additional footer text (e.g. the TP-episode counts).
    """
    parts = [f"data: {data_origin}", f"aggregation path: {path}"]
    if extra:
        parts.append(extra)
    fig.text(0.01, -0.02, "  ·  ".join(parts), ha="left", va="top",
             fontsize=7, color=INK_MUTED)
    for key, banner in _BANNERS.items():
        if data_origin.startswith(key):
            fig.text(0.99, -0.02, banner, ha="right", va="top", fontsize=8,
                     fontweight="bold", color="#b00020")


def save(fig, name: str, caption: str, out_dir: Path = FIG_DIR) -> list[Path]:
    """Write ``name``.png (300 dpi), ``name``.pdf and ``name``.caption.txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    for path in paths:
        fig.savefig(path)
    caption_path = out_dir / f"{name}.caption.txt"
    caption_path.write_text(caption.strip() + "\n", encoding="utf-8")
    plt.close(fig)
    return [*paths, caption_path]
