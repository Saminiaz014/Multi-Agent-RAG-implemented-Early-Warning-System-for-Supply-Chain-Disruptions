"""Write ``eval/RESULTS_FROZEN.md``: the results the thesis reports, every number generated.

One page, built only from committed files; no number in it is typed::

    python scripts/make_results_frozen.py           # write eval/RESULTS_FROZEN.md
    python scripts/make_results_frozen.py --check   # exit 1 if the page is out of date

Tables (each one a function of the committed CSVs / JSON, see ``INPUTS``):

* main results: tier AUC and the paired ΔAUC against Tier 1 (``results_main.csv``),
* Eq. 1 ablation: the same tiers with the duration term off, with the paired change,
* agent ablation: each agent alone and with each agent left out (paired change),
* split sensitivity: Tier 1 and Tier 4 at three split dates,
* Malacca false-alarm rate (no positives: no AUC),
* surrogate fidelity (held-out R²) and the Bab el-Mandeb diagnostic,
* lead time per onset, the random control included.

Provenance block: the commit checked out when the page was generated, each input's
git blob id and sha256, the sha256 of ``outputs/MANIFEST.json``, and the result of
checking that manifest's blob ids and hashes against git. An input must be tracked
and identical to the index (staged for the commit that adds this page, or
committed); anything else stops the run, so the page cannot describe a file nobody
can recover. The page is committed after the commit it names (it cannot contain its
own), so rerunning the script at the release tag rewrites the commit cells;
``--check`` ignores them and compares everything else, the hashes included.

Text files are hashed with LF line endings, as git stores them, so a hash taken on
Windows verifies on any clone.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "RESULTS_FROZEN.md"

#: Every file the page is built from; all must be committed (or staged) and unchanged.
INPUTS: tuple[str, ...] = (
    "eval/results_main.csv",
    "eval/method_comparison_results.csv",
    "eval/method_comparison_no_eq1.csv",
    "eval/agent_ablation.csv",
    "eval/agent_ablation_no_eq1.csv",
    "eval/split_sensitivity.csv",
    "eval/lead_time.csv",
    "eval/bem_surrogate.csv",
    "outputs/explanations_summary.json",
    "outputs/MANIFEST.json",
)

REGIONS = ("hormuz", "bab_el_mandeb", "panama", "malacca")
ABLATION_ORDER = ("full", "single", "leave-one-out")
_DASH = "—"
_PROVENANCE = "## Provenance"
_NEW = "added by the commit that contains this page"


# --------------------------------------------------------------------- sources
def load_sources(root: Path = ROOT) -> dict:
    """Read every input into the structures the tables take."""
    results = pd.read_csv(root / "eval" / "method_comparison_results.csv")
    boot = {"ci": float(results["bootstrap_ci"].dropna().iloc[0]),
            "block": int(results["bootstrap_block"].dropna().iloc[0]),
            "n": int(results["bootstrap_n"].dropna().iloc[0])}
    bem = pd.read_csv(root / "eval" / "bem_surrogate.csv")
    bem["feature"] = bem["feature"].fillna("")
    return {
        "results_main": pd.read_csv(root / "eval" / "results_main.csv"),
        "no_eq1": pd.read_csv(root / "eval" / "method_comparison_no_eq1.csv"),
        "ablation_on": pd.read_csv(root / "eval" / "agent_ablation.csv"),
        "ablation_off": pd.read_csv(root / "eval" / "agent_ablation_no_eq1.csv"),
        "split": pd.read_csv(root / "eval" / "split_sensitivity.csv"),
        "lead": pd.read_csv(root / "eval" / "lead_time.csv"),
        "bem": bem,
        "surrogate": json.loads((root / "outputs" / "explanations_summary.json")
                                .read_text(encoding="utf-8")),
        "bootstrap": boot,
    }


def _module(name: str, rel: str):
    """A sibling script loaded once by path (registered, so its dataclasses resolve)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _results_table():
    """``scripts/results_table.py``: the one place a "best" method is named."""
    return _module("results_table", "scripts/results_table.py")


def _lead_window() -> int:
    """The lead-time window (days either side of an onset), from the figure script."""
    return int(_module("detection_figures", "eval/ch4/detection_figures.py").WINDOW)


# ------------------------------------------------------------------ formatting
def _f(x) -> str:
    return _DASH if pd.isna(x) else f"{x:.3f}"


def _signed(x) -> str:
    """Signed, 3 decimals; a nonzero value that would print as zero gets 4."""
    if pd.isna(x):
        return _DASH
    return f"{x:+.4f}" if x != 0 and abs(x) < 5e-4 else f"{x:+.3f}"


def _ci(row: pd.Series, col: str = "auc") -> str:
    if pd.isna(row[col]):
        return _DASH
    return f"{_f(row[col])} [{_f(row[f'{col}_ci_low'])}, {_f(row[f'{col}_ci_high'])}]"


def _paired(row: pd.Series, col: str) -> str:
    """A paired difference with its interval, from the CSV; bold when the interval excludes zero."""
    valid = row.get(col.replace("_auc", "_valid"))
    if pd.isna(row.get(col)) or (pd.notna(valid) and not bool(valid)):
        return _DASH
    lo, hi = row[f"{col}_ci_low"], row[f"{col}_ci_high"]
    cell = f"{_signed(row[col])} [{_signed(lo)}, {_signed(hi)}]"
    return f"**{cell}**" if hi < 0 or lo > 0 else cell


def _int(x) -> str:
    return _DASH if pd.isna(x) else f"{int(x)}"


def _md(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _tier_number(method: str) -> int:
    return int(method.split()[1])


def _interval_text(boot: dict) -> str:
    return (f"{boot['ci']:.0%} moving-block bootstrap intervals (blocks of {boot['block']} days, "
            f"{boot['n']} resamples)")


# ---------------------------------------------------------------------- tables
def table_main(results: pd.DataFrame, boot: dict) -> str:
    """Tier AUC and paired ΔAUC against Tier 1, per region with positives."""
    rt = _results_table()
    lines = ["## Main results: tier AUC and paired ΔAUC against Tier 1", "",
             f"From `eval/results_main.csv`, with {_interval_text(boot)}. ΔAUC is the paired "
             "difference against Tier 1 on the same days; it is bold when its interval excludes "
             "zero. A tier that copies the tier before it (the region disables the agent it "
             "would add) is marked. † marks a model fitted on every feature column, the "
             "geopolitical ones included. Every other method is in `eval/results_main.md`."]
    for region in REGIONS:
        rows = results[results["region"] == region]
        head = rows.iloc[0]
        if rows["test_positives"].max() == 0:
            continue
        lines += ["", f"### {region}: {_int(head['test_positives'])} positives among "
                      f"{_int(head['n_test'])} scored test days "
                      f"({head['test_start']} .. {head['test_end']})", ""]
        tiers = rows[rows["row_type"] == "tier"]
        tiers = tiers.iloc[[i for _, i in sorted((_tier_number(m), k)
                                                 for k, m in enumerate(tiers["method"]))]]
        lines.append(_md(
            ["Tier", "Agents", "AUC [interval]", "ΔAUC against Tier 1 [interval]", "Alert rate",
             "Note"],
            [[r["method"], _int(r["n_agents"]), _ci(r), _paired(r, "delta_auc"),
              _f(r["alert_rate"]),
              f"copy of {r['copy_of']}" if isinstance(r.get("copy_of"), str) else ""]
             for _, r in tiers.iterrows()]))
        lines += ["", rt.best_line(rows)]
    return "\n".join(lines)


def table_eq1(results: pd.DataFrame, no_eq1: pd.DataFrame) -> str:
    """Tier AUC with the Eq. 1 duration term on (main results) and off, with the paired change."""
    on = results[results["row_type"] == "tier"].set_index(["region", "method"])
    off = no_eq1.set_index(["region", "method"])
    rows = []
    for region in REGIONS:
        for method in sorted(on.loc[region].index, key=_tier_number):
            a, b = on.loc[(region, method)], off.loc[(region, method)]
            rows.append([region, method, _ci(a), _ci(b), _paired(b, "eq1_delta_auc"),
                         _f(a["alert_rate"]), _f(b["alert_rate"])])
    return "\n".join([
        "## Eq. 1 ablation: the duration term on and off", "",
        "Tier AUC with the shipping agent's Eq. 1 duration term on (the main results) and off "
        "(`eval/method_comparison_no_eq1.csv`). The change is the paired difference, off minus "
        "on, with its interval from that file; it is bold when the interval excludes zero, so a "
        "change that is not bold cannot be told from noise.", "",
        _md(["Region", "Tier", "AUC, Eq. 1 on", "AUC, Eq. 1 off", "Paired change, off − on",
             "Alert rate, on", "Alert rate, off"], rows)])


def table_agent_ablation(on: pd.DataFrame, off: pd.DataFrame) -> str:
    """Each agent alone and the full set with each agent left out, Eq. 1 on and off."""
    off_by = off.set_index(["region", "method"])
    lines = ["## Agent ablation: single agents and leave-one-out", "",
             "From `eval/agent_ablation.csv` (Eq. 1 on) and `eval/agent_ablation_no_eq1.csv` "
             "(off). `single` is one agent alone, `leave-one-out` is the enabled set without "
             "that agent, whose paired AUC change against the full enabled set is given with its "
             "interval (bold when it excludes zero). An agent a region cannot run is listed as "
             "unavailable; a method whose scores tie at the threshold alerts on every day."]
    for region in REGIONS:
        sub = on[on["region"] == region]
        order = {k: i for i, k in enumerate(ABLATION_ORDER)}
        sub = sub.iloc[sorted(range(len(sub)), key=lambda i: (order[sub.iloc[i]["ablation"]], i))]
        rows = []
        for _, r in sub.iterrows():
            twin = off_by.loc[(region, r["method"])]
            notes = []
            if isinstance(r["agents_unavailable"], str):
                notes.append(f"unavailable: {r['agents_unavailable']}")
            if r["alert_rate"] >= 1:
                notes.append("alerts on every scored day")
            rows.append([r["ablation"], r["method"],
                         r["agents"] if isinstance(r["agents"], str) else "",
                         _ci(r), _ci(twin),
                         _paired(r, "loo_delta_auc") if r["ablation"] == "leave-one-out" else _DASH,
                         _f(r["alert_rate"]), "; ".join(notes)])
        lines += ["", f"### {region}", "",
                  _md(["Ablation", "Method", "Agents", "AUC, Eq. 1 on", "AUC, Eq. 1 off",
                       "Paired change against the full set, leave-one-out (on)",
                       "Alert rate (on)", "Note"], rows)]
    return "\n".join(lines)


def table_split(split: pd.DataFrame, headline: str | None = None) -> str:
    """Tier AUC at each split date; ``headline`` is the split the main results use."""
    methods = list(dict.fromkeys(split["method"]))
    first, last = methods[0], methods[-1]
    rows = []
    for region in REGIONS:
        for date in sorted(split["split_date"].unique()):
            part = split[(split["region"] == region) & (split["split_date"] == date)]
            by = {r["method"]: r for _, r in part.iterrows()}
            rows.append([region, f"{date} (headline split)" if str(date) == headline else date]
                        + [_ci(by[m]) for m in methods]
                        + [_signed(by[last]["auc"] - by[first]["auc"])]
                        + [_f(by[m]["alert_rate"]) for m in methods])
    headers = (["Region", "Split date"] + [f"{m} AUC" for m in methods]
               + [f"{last} − {first}"] + [f"{m} alert rate" for m in methods])
    return "\n".join([
        "## Split sensitivity", "",
        "The same methods refitted with the training/test split moved (`eval/split_sensitivity.csv`). "
        "The gap is the later tier minus the first, point estimates; a region with no positives "
        "has no AUC and shows alert rates only.", "", _md(headers, rows)])


def table_malacca(results: pd.DataFrame) -> str:
    """False-alarm and alert rates in the region without positives."""
    rt = _results_table()
    sub = results[results["region"] == "malacca"]
    head = sub.iloc[0]
    order = {m: i for i, m in enumerate(rt.METHOD_ORDER)}
    sub = sub.iloc[sorted(range(len(sub)), key=lambda i: (
        sub.iloc[i]["row_type"] == "ceiling", order.get(sub.iloc[i]["method"], len(order)), i))]
    rows = [[r["method"], r["row_type"], _f(r["far"]),
             f"{_f(r['alert_rate'])} [{_f(r['alert_rate_ci_low'])}, {_f(r['alert_rate_ci_high'])}]"]
            for _, r in sub.iterrows()]
    return "\n".join([
        "## Malacca false-alarm rate", "",
        f"malacca: {_int(head['test_positives'])} positives among {_int(head['n_test'])} scored "
        f"test days ({head['test_start']} .. {head['test_end']}). A region with no positives has "
        "no AUC: only false-alarm rates exist, and a low rate alone does not rank methods (a "
        "method that never alerts has the lowest). With no positives every alert is a false "
        "alarm, so the false-alarm rate and the alert rate coincide.", "",
        _md(["Method", "Type", "False-alarm rate", "Alert rate [interval]"], rows)])


def table_surrogate(summary: dict) -> str:
    """The SHAP surrogate's fidelity per region, from the committed explanation summary."""
    rows = []
    for region, rec in summary["regions"].items():
        rows.append([region, f"{rec['test_dates'][0]} .. {rec['test_dates'][1]}",
                     ", ".join(rec["agents"]), _f(rec["r2_test"]), _f(rec["r2_train"]),
                     _f(rec["spearman_test"]), _int(rec["alerts"]),
                     f"{_int(rec['explanations_complete'])} ({rec['complete_share']:.0%})"])
    return "\n".join([
        "## Surrogate fidelity", "",
        "How closely the random-forest surrogate reproduces the composite risk score "
        "(`outputs/explanations_summary.json`): held-out R² on the test window, R² on the "
        "training window, and the held-out Spearman rank correlation. The explanations "
        "explain the surrogate, so these bound how far they can be trusted. Alerts are the "
        "flagged days that were explained (the definition is in `outputs/explanations_summary.md`).",
        "", _md(["Region", "Test window", "Agents", "Held-out R²", "Training R²",
                 "Held-out Spearman", "Alerts", "Explanations complete"], rows)])


def table_bem(bem: pd.DataFrame, scored: float | None = None) -> str:
    """The Bab el-Mandeb surrogate diagnostic (``eval/diagnostics/bem_surrogate.py``).

    Args:
        bem: ``eval/bem_surrogate.csv``.
        scored: The scored test days the main tables report for the same region, to
            explain a difference from the surrogate's own test days.
    """
    scalar = {r["quantity"]: r["value"] for _, r in bem[bem["feature"] == ""].iterrows()}
    region = bem["region"].iloc[0]
    rows = [
        ["Test days", _int(scalar["n"])],
        ["Target mean, training → test", f"{_f(scalar['train_mean'])} → {_f(scalar['target_mean'])}"],
        ["Target shift (test mean − training mean)", _signed(scalar["target_shift"])],
        ["Var(y_test)", _f(scalar["var_target"])],
        ["MSE (held-out)", _f(scalar["mse"])],
        ["bias (mean prediction − truth)", _signed(scalar["bias"])],
        ["bias²", _f(scalar["bias_sq"])],
        ["variance of the error", _f(scalar["variance"])],
        ["bias² share of MSE", f"{scalar['bias_share']:.1%}"],
        ["R² (held-out)", _f(scalar["r2"])],
        ["R² after subtracting the mean bias", _f(scalar["r2_debiased"])],
        ["R² of predicting the training mean", _f(scalar["r2_train_mean"])],
    ]
    for feature in dict.fromkeys(bem.loc[bem["feature"] != "", "feature"]):
        f = {r["quantity"]: r["value"] for _, r in bem[bem["feature"] == feature].iterrows()}
        rows.append([f"`{feature}`: test values outside the training min–max",
                     f"{f['outside']:.1%} ({f['below']:.1%} below, {f['above']:.1%} above); "
                     f"training [{f['train_min']:.3g}, {f['train_max']:.3g}], "
                     f"test [{f['test_min']:.3g}, {f['test_max']:.3g}]"])
    rows.append(["test days with any of these features outside the training range",
                 f"{scalar['any_outside']:.1%}"])
    note = []
    if scored is not None and int(scored) != int(scalar["n"]):
        note = ["", f"The surrogate scores {_int(scalar['n'])} test days; the main tables score "
                    f"{_int(scored)} for the same region and window, because they leave out the days "
                    "masked for missing news data."]
    return "\n".join([
        "## Bab el-Mandeb surrogate diagnostic", "",
        f"How the {region} surrogate's held-out R² decomposes (`eval/bem_surrogate.csv`, from "
        "`eval/diagnostics/bem_surrogate.py`): the held-out error split into bias and variance, "
        "the R² with the mean bias removed (an upper bound on any correction, not one a model "
        "could apply in advance), and how far the top features move outside their training "
        "range. It does not explain why the target mean moved.", "",
        _md(["Quantity", "Value"], rows), *note])


def table_lead(lead: pd.DataFrame) -> str:
    """Lead time per onset for the tiers and the random control."""
    methods = list(dict.fromkeys(lead["method"]))
    window = _lead_window()
    rate_rows, onset_rows = [], []
    for region in REGIONS:
        sub = lead[lead["region"] == region]
        if sub.empty:
            continue
        for onset in sorted(sub["onset"].unique()):
            cells = []
            for m in methods:
                r = sub[(sub["method"] == m) & (sub["onset"] == onset)].iloc[0]
                cells.append("missed" if r["missed"] else
                             f"{r['lead_days']:+.0f}" + ("*" if r["censored_at_window"] else ""))
            onset_rows.append([region, onset] + cells)
        rate_rows.append([region] + [_f(sub[sub["method"] == m]["alert_rate"].iloc[0]) for m in methods])
    leads = lead["lead_days"] > 0
    verdict = ("Every method, the random control included, leads every onset" if leads.all()
               else f"{int(leads.sum())} of {len(lead)} method-onsets lead the label")
    rates = lead.drop_duplicates(["region", "method"])
    random_rate = rates.loc[rates["method"] == "M0 random", "alert_rate"].mean()
    chance = ""
    if pd.notna(random_rate):
        chance = (f" If alerts fell independently across days, a method alerting on "
                  f"{random_rate:.0%} of days, as the random control does, would have an alert in "
                  f"the {window} days before an onset with probability "
                  f"{1 - (1 - random_rate) ** window:.0%}; real alerts are autocorrelated, so that "
                  f"is an upper reference. The highest alert rate in this table is "
                  f"{rates['alert_rate'].max():.0%}.")
    return "\n".join([
        "## Lead time", "",
        f"Post hoc, from `eval/lead_time.csv` (`eval/ch4/detection_figures.py`): the lead is the "
        f"onset of a labelled run minus the first alert within {window} days either side of it, at "
        "each method's own operating point (alert when the score is at or above its "
        "training-window 90th percentile). Positive means the alert came before the onset; `*` "
        "means an alert was already active when the window opened.", "",
        f"**{verdict}.** The label is a lagged 30-day mean (its onset follows the physical drop), "
        f"and a method that alerts on many days has an alert before almost any onset.{chance} "
        "Lead time here is not evidence of early warning.", "",
        _md(["Region", "Onset"] + methods, onset_rows), "",
        "Alert rate on the scored test days:", "",
        _md(["Region"] + methods, rate_rows)])


def render_tables(src: dict) -> str:
    """Everything above the provenance block."""
    results = src["results_main"]
    bem_region = src["bem"]["region"].iloc[0]
    scored = results.loc[results["region"] == bem_region, "n_test"]
    parts = [
        "# Frozen results", "",
        "Generated by `scripts/make_results_frozen.py` from committed files; no number here is "
        "typed. Do not edit: rerun the script. The method-by-method table is "
        "[`results_main.md`](results_main.md); this page adds the ablations, the diagnostics and "
        "the provenance.", "",
        table_main(results, src["bootstrap"]), "",
        table_eq1(results, src["no_eq1"]), "",
        table_agent_ablation(src["ablation_on"], src["ablation_off"]), "",
        table_split(src["split"], headline=str(results["test_start"].iloc[0])), "",
        table_malacca(results), "",
        table_surrogate(src["surrogate"]), "",
        table_bem(src["bem"], scored=float(scored.iloc[0]) if len(scored) else None), "",
        table_lead(src["lead"]), ""]
    return "\n".join(parts)


# ------------------------------------------------------------------ provenance
def _lf(data: bytes) -> bytes:
    try:
        return data.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError:
        return data


def _blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git(root: Path, *args: str) -> tuple[int, str]:
    run = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    return run.returncode, run.stdout.strip()


def check_inputs_committed(paths, root: Path = ROOT) -> None:
    """Stop unless every input is tracked and identical to the index.

    A staged file passes (it is committed with this page); an untracked or
    locally modified one does not.

    Raises:
        ValueError: Naming each offending path.
    """
    bad = []
    for rel in paths:
        if _git(root, "ls-files", "--error-unmatch", rel)[0] != 0:
            bad.append(f"{rel} (untracked)")
        elif _git(root, "diff", "--quiet", "--", rel)[0] != 0:
            bad.append(f"{rel} (modified since it was staged or committed)")
    if bad:
        raise ValueError("inputs must be tracked and unchanged: " + ", ".join(bad))


def committed_outputs_ok(entries: list[dict], root: Path = ROOT) -> int:
    """How many committed outputs match the manifest: the file's sha256 and blob id, and HEAD's blob."""
    ok = 0
    for entry in entries:
        path = root / entry["path"]
        if not path.exists():
            continue
        data = _lf(path.read_bytes())
        head = _git(root, "rev-parse", f"HEAD:{entry['path']}")[1]
        ok += int(hashlib.sha256(data).hexdigest() == entry["sha256"]
                  and _blob(data) == entry["git_blob"] and head == entry["git_blob"])
    return ok


def provenance(root: Path = ROOT) -> dict:
    """Commit, input hashes, and the manifest's hash with its blob ids checked against git."""
    commit = _git(root, "rev-parse", "HEAD")[1]
    inputs = []
    for rel in INPUTS:
        data = _lf((root / rel).read_bytes())
        blob = _blob(data)
        in_head = _git(root, "rev-parse", f"HEAD:{rel}")[1] == blob
        last = _git(root, "log", "-1", "--format=%h", "--", rel)[1] if in_head else ""
        inputs.append({"path": rel, "blob": blob, "sha256": hashlib.sha256(data).hexdigest(),
                       "commit": last or _NEW})
    raw = _lf((root / "outputs" / "MANIFEST.json").read_bytes())
    manifest = json.loads(raw)
    mismatch = []
    for entry in manifest["code"]["files"]:
        path, recorded = entry["path"], entry["git_blob"]
        head = _git(root, "rev-parse", f"HEAD:{path}")[1]
        if head == recorded:
            continue
        holder = ""
        for candidate in _git(root, "log", "--format=%h", "--", path)[1].splitlines():
            if _git(root, "rev-parse", f"{candidate}:{path}")[1] == recorded:
                holder = candidate
                break
        mismatch.append({"path": path, "recorded": recorded, "head": head,
                         "last_commit_with_recorded": holder})
    return {"commit": commit, "inputs": inputs,
            "manifest": {"sha256": hashlib.sha256(raw).hexdigest(),
                         "generated_utc": manifest["generated_utc"],
                         "head_when_generated": manifest["code"]["head_when_generated"],
                         "code_total": len(manifest["code"]["files"]),
                         "code_uncommitted_then": len(manifest["code"].get("differs_from_head", [])),
                         "code_mismatch": mismatch,
                         "committed_total": len(manifest["committed"]),
                         "committed_ok": committed_outputs_ok(manifest["committed"], root)}}


def render_provenance(prov: dict) -> str:
    """The provenance block."""
    m = prov["manifest"]
    matched = m["code_total"] - len(m["code_mismatch"])
    lines = [
        _PROVENANCE, "",
        f"Generated with commit `{prov['commit']}` checked out; this page is committed in a later "
        "commit (a page cannot name its own), so rerunning the script at the release tag rewrites "
        "the commit cells below and leaves the working tree changed. `--check` ignores them and "
        "compares everything else. Every input was tracked and unchanged when the page was "
        "written. The blob id is git's own (sha1 over the LF content) and the sha256 is over the "
        "same bytes: to verify, `git rev-parse <commit-or-tag>:<path>` prints the blob id, and a "
        "checkout's file hashes to the sha256 once line endings are LF.", "",
        _md(["Input", "git blob", "sha256", "Last commit touching it"],
            [[f"`{i['path']}`", f"`{i['blob']}`", f"`{i['sha256']}`", i["commit"]]
             for i in prov["inputs"]]), "",
        "### Explanation manifest", "",
        f"`outputs/MANIFEST.json` has sha256 `{m['sha256']}`; it was generated {m['generated_utc']} "
        f"with `{m['head_when_generated']}` as HEAD (recorded in the manifest).", "",
        f"- Committed outputs: {m['committed_ok']} of {m['committed_total']} match their recorded "
        "sha256 and git blob id, and the blob at HEAD.",
        f"- Producing-code blob ids: {matched} of {m['code_total']} equal the blob at HEAD.",
        f"- {m['code_uncommitted_then']} of the producing files were uncommitted changes against "
        "that HEAD when the outputs were made; each is identical to HEAD now unless listed below.",
    ]
    if m["code_mismatch"]:
        lines += ["", "The others changed after the outputs were produced; the recorded blob is "
                      "the version in the commit shown, and the command prints what changed:", "",
                  _md(["Path", "Recorded blob", "Blob at HEAD", "Last commit holding the recorded blob",
                       "What changed"],
                      [[f"`{c['path']}`", f"`{c['recorded']}`", f"`{c['head']}`",
                        c["last_commit_with_recorded"] or "none: never committed",
                        (f"`git diff {c['last_commit_with_recorded']} HEAD -- {c['path']}`"
                         if c["last_commit_with_recorded"] else _DASH)]
                       for c in m["code_mismatch"]])]
    return "\n".join(lines) + "\n"


def render(root: Path = ROOT) -> str:
    """The whole page."""
    return render_tables(load_sources(root)) + "\n" + render_provenance(provenance(root))


def stale_parts(page: str, fresh_tables: str, prov: dict) -> list[str]:
    """What in ``page`` no longer matches the sources (the commit cells are ignored)."""
    problems = []
    if page.split("\n" + _PROVENANCE)[0].rstrip("\n") != fresh_tables.rstrip("\n"):
        problems.append("the tables differ from a fresh render")
    block = page[page.find(_PROVENANCE):] if _PROVENANCE in page else ""
    for item in prov["inputs"]:
        if item["blob"] not in block or item["sha256"] not in block:
            problems.append(f"{item['path']}: blob id or sha256 differs")
    if f"has sha256 `{prov['manifest']['sha256']}`" not in block:
        problems.append("the manifest hash differs")
    for entry in prov["manifest"]["code_mismatch"]:
        if entry["recorded"] not in block:
            problems.append(f"{entry['path']}: recorded blob id missing")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the page (the default)")
    mode.add_argument("--check", action="store_true",
                      help="exit 1 if the page differs from the sources (commit cells ignored)")
    args = parser.parse_args(argv)
    try:
        check_inputs_committed(INPUTS)
    except ValueError as error:
        print(error)
        return 2
    if args.check:
        if not OUT.exists():
            print(f"{OUT} does not exist: run without --check")
            return 1
        page = OUT.read_bytes().decode("utf-8").replace("\r\n", "\n")
        problems = stale_parts(page, render_tables(load_sources()), provenance())
        print("up to date" if not problems else
              f"{OUT} is out of date ({'; '.join(problems)}): rerun without --check")
        return 1 if problems else 0
    OUT.write_text(render(), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
