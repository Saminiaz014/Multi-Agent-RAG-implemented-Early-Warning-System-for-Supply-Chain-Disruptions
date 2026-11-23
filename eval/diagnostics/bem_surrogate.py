"""Why the Bab el-Mandeb surrogate has a negative held-out R² (issue 15; diagnostic, no fix).

Loads the surrogate fitted by ``scripts/run_real_surrogate.py``
(``outputs/surrogate/real_rf.joblib``), rebuilds the region's test window
exactly as that script does, and reports, in one table. The model file and the
evaluation frames are git-ignored: run that script, and the frame build, first.

* the held-out error split into bias and variance:
  ``MSE = bias² + variance``, with ``bias`` the mean prediction error and
  ``variance`` the variance of that error;
* the R² the surrogate gets, the R² after subtracting the mean bias
  (``1 - variance / Var(y_test)``), and the R² of predicting the training mean;
* for each of the region's top-5 SHAP features, the share of test values
  outside the training minimum-maximum (below, above, and either), and the
  share of test days where any of the five is outside.

It first checks that the loaded model reproduces the R² committed in
``outputs/surrogate/real_surrogate_metrics.json`` and that the top-5 features'
summary (``outputs/explanations_summary.json``) is from the same run; if not,
it stops.

Usage::

    python eval/diagnostics/bem_surrogate.py

Writes ``data/eval/diagnostics/bem_surrogate.md`` (git-ignored) and prints it, and
``eval/bem_surrogate.csv`` (committed): the same numbers in long form
(``region, quantity, feature, value``), which ``eval/RESULTS_FROZEN.md`` reads.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eval" / "diagnostics"
CSV_OUT = ROOT / "eval" / "bem_surrogate.csv"
REGION = "bab_el_mandeb"


_NOTES = (
    "Notes. The R² after subtracting the mean bias removes this test window's own bias, so it is "
    "an upper bound on any correction, not one a model could apply in advance. The table does "
    "not measure covariate shift inside the training range, does not explain why the target "
    "mean moved, and leaves the remaining error variance unexplained."
)


def error_decomposition(y_true, y_pred, train_mean: float | None = None) -> dict:
    """Bias / variance split of the prediction error, and the R² variants.

    Args:
        y_true: The held-out target.
        y_pred: The surrogate's predictions for it.
        train_mean: The training target's mean, for the constant baseline.

    Returns:
        ``mse = bias_sq + variance`` exactly (population variance), the
        held-out ``r2``, ``r2_debiased = 1 - variance / Var(y_true)`` (the R² if
        the mean error were subtracted), and with ``train_mean`` the R² of
        predicting it (``r2_train_mean``) and ``target_shift`` (test mean minus
        training mean).
    """
    y = np.asarray(y_true, dtype=float)
    error = np.asarray(y_pred, dtype=float) - y
    bias, variance = float(error.mean()), float(error.var())
    mse = float(np.mean(error ** 2))
    spread = float(y.var())
    if spread == 0.0:
        raise ValueError("the held-out target is constant: R² is undefined")
    out = {"n": int(len(y)), "mse": mse, "bias": bias, "bias_sq": bias ** 2,
           "variance": variance, "bias_share": bias ** 2 / mse if mse else float("nan"),
           "var_target": spread, "r2": 1.0 - mse / spread,
           "r2_debiased": 1.0 - variance / spread, "target_mean": float(y.mean())}
    if train_mean is not None:
        out["target_shift"] = float(y.mean()) - float(train_mean)
        out["r2_train_mean"] = 1.0 - float(np.mean((y - train_mean) ** 2)) / spread
        out["train_mean"] = float(train_mean)
    return out


def outside_range(train: pd.Series, test: pd.Series) -> dict:
    """Shares of test values below, above and outside the training min-max (NaN skipped)."""
    if train.notna().sum() == 0:
        raise ValueError("the training column has no values: its range is undefined")
    lo, hi = float(train.min()), float(train.max())
    values = test.dropna()
    below, above = float((values < lo).mean()), float((values > hi).mean())
    return {"train_min": lo, "train_max": hi, "test_min": float(values.min()),
            "test_max": float(values.max()), "below": below, "above": above,
            "outside": below + above, "n_test": int(len(values))}


def any_outside(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> float:
    """Share of test days on which at least one of ``features`` is outside the training range."""
    out = pd.Series(False, index=test.index)
    for f in features:
        if train[f].notna().sum() == 0:
            raise ValueError(f"{f}: the training column has no values: its range is undefined")
        out |= (test[f] < train[f].min()) | (test[f] > train[f].max())   # NaN counts as inside
    return float(out.mean())


def render_table(decomp: dict, train: pd.DataFrame, test: pd.DataFrame,
                 features: list[str]) -> str:
    """The single diagnostic table, as markdown."""
    rows = [
        ("Test days", f"{decomp['n']}"),
        ("Target mean, training → test", f"{decomp.get('train_mean', float('nan')):.3f} → "
                                          f"{decomp['target_mean']:.3f}"),
        ("Var(y_test)", f"{decomp['var_target']:.4f}"),
        ("MSE (held-out)", f"{decomp['mse']:.4f}"),
        ("bias (mean prediction − truth)", f"{decomp['bias']:+.4f}"),
        ("bias²", f"{decomp['bias_sq']:.4f}"),
        ("variance of the error", f"{decomp['variance']:.4f}"),
        ("bias² share of MSE", f"{decomp['bias_share']:.1%}"),
        ("R² (held-out)", f"{decomp['r2']:.3f}"),
        ("R² after subtracting the mean bias", f"{decomp['r2_debiased']:.3f}"),
        ("R² of predicting the training mean", f"{decomp.get('r2_train_mean', float('nan')):.3f}"),
    ]
    for f in features:
        r = outside_range(train[f], test[f])
        rows.append((f"`{f}`: test values outside the training min–max",
                     f"{r['outside']:.1%} ({r['below']:.1%} below, {r['above']:.1%} above); "
                     f"train [{r['train_min']:.3g}, {r['train_max']:.3g}], "
                     f"test [{r['test_min']:.3g}, {r['test_max']:.3g}]"))
    rows.append(("test days with any of these five outside the training range",
                 f"{any_outside(train, test, features):.1%}"))
    table = "\n".join(["| Quantity | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows])
    return table + "\n\n" + _NOTES + "\n"


def numeric_rows(decomp: dict, train: pd.DataFrame, test: pd.DataFrame,
                 features: list[str]) -> list[dict]:
    """The diagnostic's numbers as long-form rows (``quantity``, ``feature``, ``value``).

    ``feature`` is empty for the error decomposition and ``any_outside``, and
    names the feature for each :func:`outside_range` quantity.
    """
    rows = [{"quantity": k, "feature": "", "value": float(v)} for k, v in decomp.items()]
    rows.append({"quantity": "any_outside", "feature": "",
                 "value": any_outside(train, test, features)})
    for f in features:
        for k, v in outside_range(train[f], test[f]).items():
            rows.append({"quantity": k, "feature": f, "value": float(v)})
    return rows


def write_rows(rows: list[dict], path: Path, region: str = REGION) -> None:
    """Write :func:`numeric_rows` to ``path`` as ``region, quantity, feature, value``."""
    frame = pd.DataFrame(rows)
    frame.insert(0, "region", region)
    frame.to_csv(path, index=False, lineterminator="\n")


def _load_script(name: str, rel: str):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import joblib
    from sklearn.metrics import r2_score

    rrs = _load_script("run_real_surrogate", "scripts/run_real_surrogate.py")
    bundle = joblib.load(ROOT / "outputs" / "surrogate" / "real_rf.joblib")
    model, columns = bundle["models"][REGION], bundle["features"][REGION]
    d = rrs.prepare_region(REGION)
    X_tr, y_tr, X_te, y_te, _, _ = rrs._parts(d, columns)
    pred = model.predict(X_te)

    committed = json.loads((ROOT / "outputs" / "surrogate" / "real_surrogate_metrics.json")
                           .read_text(encoding="utf-8"))["results"][REGION]["metrics"]["r2_test"]
    if abs(r2_score(y_te, pred) - committed) > 1e-6:
        print(f"STOP: the loaded model gives R² {r2_score(y_te, pred):.6f}, the committed metrics "
              f"say {committed:.6f}; the model and the data do not match.")
        return 1
    summary = json.loads((ROOT / "outputs" / "explanations_summary.json").read_text(encoding="utf-8"))
    if abs(summary["regions"][REGION]["r2_test"] - committed) > 1e-6:
        print("STOP: outputs/explanations_summary.json is from a different run than the "
              "surrogate metrics, so its top-5 features may not belong to this model.")
        return 1
    top = [t["feature"] for t in summary["regions"][REGION]["top_shap_features"]]
    decomp = error_decomposition(y_te, pred, train_mean=float(np.mean(y_tr)))
    table = render_table(decomp, X_tr, X_te, top)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bem_surrogate.md").write_text(table, encoding="utf-8")
    write_rows(numeric_rows(decomp, X_tr, X_te, top), CSV_OUT)
    print(f"reproduces the committed held-out R² ({committed:.3f})\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
