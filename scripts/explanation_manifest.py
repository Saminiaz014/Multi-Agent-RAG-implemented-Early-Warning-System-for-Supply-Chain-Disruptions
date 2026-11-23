"""Summary and manifest of the explanation outputs (issue 13).

Run after ``scripts/run_real_surrogate.py`` and ``scripts/run_alert_explanations.py``.
It makes the thesis values traceable without committing the large artefacts:

* ``outputs/explanations_summary.md`` / ``.json`` (committed): per region, the
  surrogate's fit (held-out and training R², Spearman), the test window, the
  alert count and explanation completeness, and the top 5 SHAP features.
* ``outputs/MANIFEST.json`` (committed): the provenance of those values --

  - **code**: the git blob id of every producing script, module and config
    file, so ``git ls-tree <commit> <path>`` shows which commit holds exactly
    that code (and which files differed from HEAD when the outputs were made);
  - **inputs**: sha256 of the evaluation frames and their manifests
    (``data/eval/``, git-ignored), and each region's ACLED fetch (row counts per
    country-year, pages, a sha256 of the sorted event ids; ``acled_events``);
  - **artefacts**: size and sha256 of the git-ignored outputs -- the fitted
    surrogates (``outputs/surrogate/*.joblib``) and the per-alert explanation
    files (``outputs/explanations/``, digested as one tree);
  - **committed**: the blob id and sha256 of every committed output;
  - library versions.

Text files are hashed with LF line endings (``\\r\\n`` -> ``\\n``), as git
stores them, so a hash taken on Windows verifies on any clone. Blob ids are
git's own (``sha1("blob <n>\\0" + content)`` over the LF content).

Top SHAP features: the exact mean |SHAP| over every feature of every alert
(``coverage.json`` ``mean_abs_shap``). ``top5_from_alert_files`` repeats the
estimate from the per-alert files, which keep only their five largest (absent
counts 0); it is a lower bound, kept because older runs only have those files.

Usage::

    python scripts/explanation_manifest.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUTS = _PROJECT_ROOT / "outputs"
_SURROGATE = _OUTPUTS / "surrogate"
_EXPLANATIONS = _OUTPUTS / "explanations"
#: Committed alongside the manifest; kept out of the tree digest.
_COMMITTED_IN_TREE = {"coverage.json"}
#: Hashed with LF line endings.
_TEXT = {".json", ".md", ".csv", ".py", ".yaml", ".yml", ".txt"}
#: The code and config that produce the outputs.
PRODUCING_CODE: tuple[str, ...] = (
    "scripts/run_real_surrogate.py", "scripts/run_alert_explanations.py",
    "scripts/run_method_comparison.py", "scripts/explanation_manifest.py",
    "src/explainability/*.py", "src/evaluation/*.py", "src/aggregation/*.py",
    "src/agents/*.py", "src/core/*.py", "src/ingestion/geopolitical_connector.py",
    "src/extractors/acled_extractor.py", "scripts/build_eval_dataset.py",
    "config/settings.yaml", "config/regions/*.yaml",
)
_LIBRARIES = ("numpy", "pandas", "scikit-learn", "shap", "joblib")


def content(path: Path) -> bytes:
    """File bytes, with CRLF -> LF for text files."""
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix.lower() in _TEXT else data


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_entry(path: Path, root: Path = _PROJECT_ROOT) -> dict:
    data = content(path)
    return {"path": path.relative_to(root).as_posix(), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "git_blob": git_blob_id(data)}


def tree_digest(folder: Path, exclude: set[str] = frozenset()) -> dict:
    """One sha256 over a folder: sorted ``name sha256`` lines of its files."""
    lines, total = [], 0
    for f in sorted(p for p in folder.iterdir() if p.is_file() and p.name not in exclude):
        data = content(f)
        total += len(data)
        lines.append(f"{f.name} {hashlib.sha256(data).hexdigest()}")
    return {"files": len(lines), "bytes": total,
            "sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()}


def top_features_from_alerts(folder: Path, region: str, k: int = 5) -> list[dict]:
    """Mean |SHAP| from the per-alert files (top five each; absent = 0): a lower bound."""
    alerts = [json.loads(p.read_text(encoding="utf-8"))
              for p in sorted(folder.glob(f"{region}_*.json"))]
    total: dict[str, float] = defaultdict(float)
    leads: dict[str, int] = defaultdict(int)
    for alert in alerts:
        ranked = sorted(alert["features"], key=lambda f: -abs(f["shap_value"]))
        for feature in ranked:
            total[feature["feature"]] += abs(feature["shap_value"])
        if ranked:
            leads[ranked[0]["feature"]] += 1
    n = max(len(alerts), 1)
    order = sorted(total, key=lambda f: -total[f])[:k]
    return [{"feature": f, "mean_abs_shap": total[f] / n, "top1_share": leads[f] / n}
            for f in order]


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=_PROJECT_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def code_provenance() -> dict:
    head = _git("rev-parse", "HEAD")
    files = sorted({p for pattern in PRODUCING_CODE for p in _PROJECT_ROOT.glob(pattern)
                    if p.is_file()})
    entries, differs = [], []
    for path in files:
        rel = path.relative_to(_PROJECT_ROOT).as_posix()
        blob = git_blob_id(content(path))
        listed = _git("ls-tree", "HEAD", "--", rel).split()
        at_head = listed[2] if len(listed) >= 3 else None
        entries.append({"path": rel, "git_blob": blob})
        if blob != at_head:
            differs.append(rel)
    return {"head_when_generated": head, "differs_from_head": differs, "files": entries,
            "verify": "git ls-tree <commit> -- <path> shows the same blob id in the commit "
                      "that holds this code"}


def input_provenance() -> list[dict]:
    data = _PROJECT_ROOT / "data" / "eval"
    return [file_entry(p) for p in sorted(data.glob("*.csv")) + sorted(data.glob("*.manifest.json"))]


def acled_provenance(eval_dir: Path | None = None) -> dict:
    """Each region's ACLED fetch: row counts per country-year and an event-id hash.

    The hash is over the sorted event ids: it pins which events were fetched,
    not their fields, which ACLED can revise under the same id.

    From the frame manifests' ``geopolitical_source`` block (``data/eval/`` is
    git-ignored, so the counts and hashes are recorded here, in the committed
    manifest). A region whose frame has ACLED features but no block -- built
    before the cursor fetch, when country-years over 5,000 events were cut --
    is named, not skipped.
    """
    folder = eval_dir if eval_dir is not None else _PROJECT_ROOT / "data" / "eval"
    out: dict = {}
    for path in sorted(folder.glob("*.manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if "geopolitical" not in (manifest.get("domains_present") or {}):
            continue
        source = manifest.get("geopolitical_source")
        if not source:
            out[path.name.removesuffix(".manifest.json")] = {"geopolitical_source": None}
            continue
        out[path.name.removesuffix(".manifest.json")] = {
            "frame_manifest_sha256": file_entry(path, root=folder)["sha256"],
            "page_size": source["page_size"], "years": source.get("years"),
            "events_total": source["events_total"], "events_sha256": source["events_sha256"],
            "country_years": source["country_years"],
            "round_cap_country_years": source["round_cap_country_years"],
        }
    return out


def versions() -> dict:
    out = {"python": platform.python_version()}
    for lib in _LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


def summary() -> dict:
    metrics = json.loads((_SURROGATE / "real_surrogate_metrics.json").read_text(encoding="utf-8"))
    coverage = {c["region"]: c for c in
                json.loads((_EXPLANATIONS / "coverage.json").read_text(encoding="utf-8"))}
    regions = {}
    for region, result in metrics["results"].items():          # the metrics file's order
        if region not in coverage:
            continue                                            # the pooled model
        m, cov = result["metrics"], coverage[region]
        exact = cov.get("mean_abs_shap") or {}
        regions[region] = {
            "r2_test": m["r2_test"], "r2_train": m["r2_train"],
            "spearman_test": m["spearman_test"], "test_dates": result["test_dates"],
            "agents": cov["agents"], "alerts": cov["alerts"],
            "explanations_complete": cov["complete"], "complete_share": cov["coverage"],
            "top_shap_features": [{"feature": f, "mean_abs_shap": v}
                                  for f, v in list(exact.items())[:5]],
            "top5_from_alert_files": top_features_from_alerts(_EXPLANATIONS, region),
        }
    return {"split": metrics["split"], "target": metrics["target"],
            "top_feature_definition": "exact mean |SHAP| over every feature of every alert "
            "(coverage.json mean_abs_shap)", "regions": regions}


def markdown(s: dict) -> str:
    out = ["# Alert explanations: summary", "",
           f"Surrogate target: {s['target']}. Split: {s['split']}.",
           f"Top SHAP features: {s['top_feature_definition']}.",
           "Complete: the explanation found history and a z-score for each of its top five "
           "features. Provenance (code, inputs, artefact hashes): `outputs/MANIFEST.json`.", "",
           "| Region | Test window | R² test | R² train | Spearman test | Alerts | Complete | "
           "Top 5 SHAP features (mean abs SHAP) |",
           "|---|---|---|---|---|---|---|---|"]
    for region, r in s["regions"].items():
        top = ", ".join(f"{t['feature']} ({t['mean_abs_shap']:.3f})" for t in r["top_shap_features"])
        share = "–" if r["complete_share"] is None else f"{r['complete_share']:.1%}"
        out.append(f"| {region} | {r['test_dates'][0]} .. {r['test_dates'][1]} | "
                   f"{r['r2_test']:.3f} | {r['r2_train']:.3f} | {r['spearman_test']:.3f} | "
                   f"{r['alerts']} | {share} | {top} |")
    poor = [f"{region} (held-out R² {r['r2_test']:.3f})" for region, r in s["regions"].items()
            if r["r2_test"] < 0]
    if poor:
        out += ["", "A negative held-out R² means the surrogate predicts the composite worse "
                "than its mean on the test window, so its SHAP values explain the surrogate "
                "more than the composite: " + ", ".join(poor) + "."]
    return "\n".join(out) + "\n"


def manifest() -> dict:
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "hashing": "sha256 and git blob ids over LF-normalised text (binary files as is)",
        "produced_by": ["scripts/run_real_surrogate.py", "scripts/run_alert_explanations.py",
                        "scripts/explanation_manifest.py"],
        "code": code_provenance(),
        "inputs": input_provenance(),
        "acled_events": acled_provenance(),
        "versions": versions(),
        "git_ignored_artefacts": {
            "files": [file_entry(p) for p in sorted(_SURROGATE.glob("*.joblib"))],
            "trees": {"outputs/explanations/": tree_digest(_EXPLANATIONS, _COMMITTED_IN_TREE)},
        },
        "committed": [file_entry(p) for p in (
            _SURROGATE / "real_surrogate_metrics.md", _SURROGATE / "real_surrogate_metrics.json",
            _SURROGATE / "synthetic_surrogate_metrics.json", _EXPLANATIONS / "coverage.json",
            _OUTPUTS / "explanations_summary.md", _OUTPUTS / "explanations_summary.json")],
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    s = summary()
    (_OUTPUTS / "explanations_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (_OUTPUTS / "explanations_summary.md").write_text(markdown(s), encoding="utf-8")
    (_OUTPUTS / "MANIFEST.json").write_text(json.dumps(manifest(), indent=2), encoding="utf-8")
    print(markdown(s))
    print(f"wrote {_OUTPUTS / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
