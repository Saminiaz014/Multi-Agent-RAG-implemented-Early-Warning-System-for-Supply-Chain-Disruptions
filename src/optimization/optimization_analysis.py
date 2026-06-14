"""Post-optimization analysis and visualisation.

Renders, as static PNGs under ``data/processed/``:

1. Optuna's built-in diagnostics — optimization history, hyper-parameter
   importances, parallel-coordinate, and a contour of the two most important
   parameters.
2. A hand-tuned vs optimized **weight** comparison (Layer 2 inter-agent
   weights plus every agent's Layer 1 intra-agent weights), so the reader can
   see at a glance which weights moved and by how much.
3. A hand-tuned vs optimized **performance** comparison (F1 / lead-time /
   FPR on both validation and test).

Everything is exported via Plotly + Kaleido. Each plot is guarded so a single
rendering failure (e.g. a degenerate study with too few trials for importance
analysis) never aborts the rest of the report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.optimization.weight_config import resolve_active_weights

logger = logging.getLogger(__name__)

_HAND_COLOR = "#9aa5b1"
_OPT_COLOR = "#2b6cb0"

_INTRA_AGENTS = (
    "shipping",
    "market",
    "geopolitical",
    "natural_disaster",
    "routing",
    "news_sentiment",
)


class OptimizationAnalysis:
    """Generate the optimization report figures.

    Args:
        study: The completed Optuna study.
        results: The results dict from
            :meth:`~src.optimization.weight_optimizer.WeightOptimizer.optimize`.
        config: Full application config (used to recover the hand-tuned
            weights for the comparison charts).
        output_dir: Directory the PNGs are written to.
    """

    def __init__(
        self,
        study: optuna.Study,
        results: dict,
        config: dict,
        output_dir: str | Path = "data/processed",
    ) -> None:
        self.study = study
        self.results = results
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def generate_all(self) -> list[Path]:
        """Render every figure; return the list of files actually written."""
        written: list[Path] = []
        for fn in (
            self._optuna_history,
            self._optuna_param_importances,
            self._optuna_parallel_coordinate,
            self._optuna_contour,
            self._weight_comparison,
            self._performance_comparison,
        ):
            try:
                path = fn()
                if path is not None:
                    written.append(path)
            except Exception as exc:  # pragma: no cover - rendering is best-effort
                logger.warning("[OptimizationAnalysis] %s failed: %s", fn.__name__, exc)
        logger.info("[OptimizationAnalysis] wrote %d figures to %s",
                    len(written), self.output_dir)
        return written

    # ------------------------------------------------------------------
    # Optuna built-ins
    # ------------------------------------------------------------------
    def _write(self, fig: go.Figure, name: str) -> Path:
        path = self.output_dir / name
        fig.write_image(str(path), width=1000, height=600, scale=2)
        return path

    def _optuna_history(self) -> Path:
        fig = optuna.visualization.plot_optimization_history(self.study)
        return self._write(fig, "optimization_history.png")

    def _optuna_param_importances(self) -> Path | None:
        completed = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed) < 2:
            logger.info("[OptimizationAnalysis] too few trials for importances.")
            return None
        fig = optuna.visualization.plot_param_importances(self.study)
        return self._write(fig, "param_importances.png")

    def _optuna_parallel_coordinate(self) -> Path:
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        return self._write(fig, "parallel_coordinate.png")

    def _optuna_contour(self) -> Path | None:
        # Contour the two most important parameters when importances are
        # computable, else fall back to Optuna's default pair.
        params = None
        try:
            importances = optuna.importance.get_param_importances(self.study)
            top = list(importances.keys())[:2]
            if len(top) == 2:
                params = top
        except Exception:  # pragma: no cover - importance can fail on tiny studies
            params = None
        fig = optuna.visualization.plot_contour(self.study, params=params)
        return self._write(fig, "contour_plot.png")

    # ------------------------------------------------------------------
    # Weight comparison
    # ------------------------------------------------------------------
    def _weight_comparison(self) -> Path:
        hand = resolve_active_weights({**self.config, "weight_mode": "hand_tuned"})
        best = self.results["best_weights"]

        titles = ["Inter-agent (Layer 2)"] + [
            a.replace("_", " ").title() for a in _INTRA_AGENTS
        ]
        fig = make_subplots(rows=4, cols=2, subplot_titles=titles)

        panels: list[tuple[dict, dict]] = [
            (hand["inter_agent_weights"], best["inter_agent_weights"])
        ]
        for agent in _INTRA_AGENTS:
            panels.append(
                (
                    hand["intra_agent_weights"].get(agent, {}),
                    best["intra_agent_weights"].get(agent, {}),
                )
            )

        for i, (hand_w, opt_w) in enumerate(panels):
            row = i // 2 + 1
            col = i % 2 + 1
            keys = list(hand_w.keys()) or list(opt_w.keys())
            short = [k.replace("_", " ")[:10] for k in keys]
            fig.add_trace(
                go.Bar(
                    x=short, y=[hand_w.get(k, 0.0) for k in keys],
                    name="Hand-tuned", marker_color=_HAND_COLOR,
                    showlegend=(i == 0),
                ),
                row=row, col=col,
            )
            fig.add_trace(
                go.Bar(
                    x=short, y=[opt_w.get(k, 0.0) for k in keys],
                    name="Optimized", marker_color=_OPT_COLOR,
                    showlegend=(i == 0),
                ),
                row=row, col=col,
            )

        fig.update_layout(
            title="Weight Comparison: Hand-Tuned vs Optimized",
            barmode="group", height=1100, width=1000,
            legend=dict(orientation="h", y=1.05),
        )
        return self._write_sized(fig, "weight_comparison.png", 1000, 1100)

    # ------------------------------------------------------------------
    # Performance comparison
    # ------------------------------------------------------------------
    def _performance_comparison(self) -> Path:
        hand = self.results["hand_tuned_metrics"]
        opt_val = self.results["validation_metrics"]
        opt_test = self.results["test_metrics"]

        groups = [
            ("F1 (val)", hand["validation"]["f1"], opt_val["f1"]),
            ("F1 (test)", hand["test"]["f1"], opt_test["f1"]),
            ("Lead score (val)", hand["validation"]["lead_time_score"],
             opt_val["lead_time_score"]),
            ("Lead score (test)", hand["test"]["lead_time_score"],
             opt_test["lead_time_score"]),
            ("FPR (val)", hand["validation"]["fpr"], opt_val["fpr"]),
            ("FPR (test)", hand["test"]["fpr"], opt_test["fpr"]),
        ]
        labels = [g[0] for g in groups]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=[g[1] for g in groups],
            name="Hand-tuned", marker_color=_HAND_COLOR,
            text=[f"{g[1]:.2f}" for g in groups], textposition="auto",
        ))
        fig.add_trace(go.Bar(
            x=labels, y=[g[2] for g in groups],
            name="Optimized", marker_color=_OPT_COLOR,
            text=[f"{g[2]:.2f}" for g in groups], textposition="auto",
        ))
        fig.update_layout(
            title="Performance: Hand-Tuned vs Optimized (validation + test)",
            barmode="group", yaxis_title="Score",
            legend=dict(orientation="h", y=1.05),
        )
        return self._write(fig, "performance_comparison.png")

    def _write_sized(self, fig: go.Figure, name: str, w: int, h: int) -> Path:
        path = self.output_dir / name
        fig.write_image(str(path), width=w, height=h, scale=2)
        return path


def generate_optimization_report(
    optimizer, results: dict, output_dir: str | Path = "data/processed"
) -> list[Path]:
    """Convenience wrapper: render all figures from an optimizer + its results.

    Args:
        optimizer: A :class:`~src.optimization.weight_optimizer.WeightOptimizer`
            whose :meth:`optimize` has already run (exposes ``study`` + ``config``).
        results: The results dict returned by ``optimizer.optimize``.
        output_dir: Output directory for the PNGs.

    Returns:
        Paths of the figures that were successfully written.
    """
    analysis = OptimizationAnalysis(
        optimizer.study, results, optimizer.config, output_dir
    )
    return analysis.generate_all()
