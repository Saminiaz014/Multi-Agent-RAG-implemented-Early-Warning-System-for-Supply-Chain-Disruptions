"""scripts/make_results_frozen.py: eval/RESULTS_FROZEN.md, every number generated from committed files."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rf = _load("make_results_frozen", "scripts/make_results_frozen.py")
SRC = rf.load_sources()
DOC = (_ROOT / "eval" / "RESULTS_FROZEN.md")
needs_git = pytest.mark.skipif(not (_ROOT / ".git").exists(),
                               reason="needs the repository's git history")
_DECIMAL = re.compile(r"[-+]?\d+\.\d+")                      # any printed decimal, 0.16 as 0.159


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True).stdout.strip()


def _section(text: str, title: str) -> str:
    start = text.index(f"## {title}")
    nxt = text.find("\n## ", start + 1)
    return text[start:] if nxt < 0 else text[start:nxt]


# ------------------------------------------------------------ the file
def test_the_file_is_the_generated_tables_then_the_provenance() -> None:
    text = DOC.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert text.split("\n## Provenance")[0].rstrip("\n") == rf.render_tables(SRC).rstrip("\n")
    assert "\n## Provenance" in text


def test_every_requested_section_is_present_in_order() -> None:
    text = rf.render_tables(SRC)
    titles = ["Main results", "Eq. 1 ablation", "Agent ablation", "Split sensitivity",
              "Malacca false-alarm rate", "Surrogate fidelity", "Bab el-Mandeb surrogate diagnostic",
              "Lead time"]
    positions = [text.index(f"## {t}") for t in titles]
    assert positions == sorted(positions)


# ------------------------------------------------ every number is generated
def _perturbed(src: dict) -> dict:
    """The same sources with every measured number moved to a different value."""
    out = copy.deepcopy(src)
    skip = {"n_agents", "test_positives", "n_test", "bootstrap_valid", "bootstrap_block",
            "bootstrap_n", "bootstrap_ci", "lead_days", "n"}
    for key, value in out.items():
        if isinstance(value, pd.DataFrame):
            for col in value.columns:
                if col in skip or not pd.api.types.is_float_dtype(value[col]):
                    continue
                # fixed point far outside any metric; an exact 0 or 1 prints as an integer, so it
                # stays one (shifting it would add a decimal and change the table's shape)
                moved = value[col] * 1.1 + 0.0371
                value[col] = moved.where(~value[col].isin([0.0, 1.0]), value[col])
            if key == "bem":                                   # integer ranges (counts) stay integers
                ends = value["quantity"].isin(["train_min", "train_max", "test_min", "test_max"])
                whole = ends & (src[key]["value"] % 1 == 0)
                value.loc[whole, "value"] = src[key].loc[whole, "value"]
        elif key == "surrogate":
            for region in value["regions"].values():
                for k in ("r2_test", "r2_train", "spearman_test"):
                    region[k] = region[k] * 1.1 + 0.0371
    return out


def test_every_decimal_in_the_tables_changes_when_the_sources_change() -> None:
    """Position by position: the k-th number printed must follow the k-th number read.

    A set comparison would trip on chance collisions between different source values; a
    printed number that does not move when its source moves is typed or constant.
    """
    before = _DECIMAL.findall(rf.render_tables(SRC))
    after = _DECIMAL.findall(rf.render_tables(_perturbed(SRC)))
    assert len(before) > 400                                  # the tables carry real content
    assert len(before) == len(after)
    frozen = [(i, a) for i, (a, b) in enumerate(zip(before, after))
              if a == b and abs(float(a)) >= 0.0005           # a zero difference prints alike
              and float(a) != 1.0]                            # an exact 1 was left unperturbed
    assert not frozen, frozen[:10]


def test_the_script_types_no_result_number() -> None:
    source = (_ROOT / "scripts" / "make_results_frozen.py").read_text(encoding="utf-8")
    assert not re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", source.replace("sha1", "")), \
        re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", source)


# ------------------------------------------------ cells against the CSVs
def _row(df: pd.DataFrame, **where) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for k, v in where.items():
        mask &= df[k] == v
    assert mask.sum() == 1, where
    return df[mask].iloc[0]


def test_main_table_tier_cells_are_the_csv_values() -> None:
    section = _section(rf.render_tables(SRC), "Main results")
    for region in ("hormuz", "bab_el_mandeb", "panama"):
        for tier in ("Tier 1", "Tier 5"):
            r = _row(SRC["results_main"], region=region, method=tier)
            assert f"{r['auc']:.3f}" in section and f"{r['auc_ci_low']:.3f}" in section
            if tier == "Tier 5":
                assert f"{r['delta_auc']:+.3f}" in section, (region, tier)
    assert "Best non-ceiling method" in section
    assert "ORACLE" not in section                           # a ceiling row, never ranked here


def test_eq1_table_gives_the_paired_change_with_its_interval() -> None:
    section = _section(rf.render_tables(SRC), "Eq. 1 ablation")
    on = _row(SRC["results_main"], region="panama", method="Tier 4")
    off = _row(SRC["no_eq1"], region="panama", method="Tier 4")
    line = next(ln for ln in section.splitlines() if ln.startswith("| panama | Tier 4 |"))
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    assert cells[2].startswith(f"{on['auc']:.3f}") and cells[3].startswith(f"{off['auc']:.3f}")
    d, lo, hi = off["eq1_delta_auc"], off["eq1_delta_auc_ci_low"], off["eq1_delta_auc_ci_high"]
    assert f"{rf._signed(d)} [{rf._signed(lo)}, {rf._signed(hi)}]" in cells[4]
    assert (cells[4].startswith("**")) == bool(hi < 0 or lo > 0)     # bold exactly when clear
    assert cells[5] == f"{on['alert_rate']:.3f}" and cells[6] == f"{off['alert_rate']:.3f}"


def test_the_csv_eq1_delta_is_off_minus_on_so_the_header_is_true() -> None:
    for _, off in SRC["no_eq1"].iterrows():
        if pd.isna(off["auc"]):
            continue
        on = _row(SRC["results_main"], region=off["region"], method=off["method"])
        assert off["eq1_delta_auc"] == pytest.approx(off["auc"] - on["auc"], abs=1e-9), \
            (off["region"], off["method"])


def test_agent_ablation_lists_every_single_and_leave_one_out_row() -> None:
    section = _section(rf.render_tables(SRC), "Agent ablation")
    on = SRC["ablation_on"]
    for region in on["region"].unique():
        n_rows = int((on["region"] == region).sum())
        block = section.split(f"### {region}")[1].split("###")[0]
        assert block.count("\n| ") - 1 >= n_rows              # header row excluded
    assert "single" in section and "leave-one-out" in section
    r = _row(on, region="panama", method="shipping alone")
    assert f"{r['auc']:.3f}" in section


def test_split_table_has_each_split_date_and_the_tier_gap() -> None:
    section = _section(rf.render_tables(SRC), "Split sensitivity")
    split = SRC["split"]
    for d in split["split_date"].unique():
        assert str(d) in section
    t1 = _row(split, region="hormuz", split_date="2024-01-01", method="Tier 1")
    t4 = _row(split, region="hormuz", split_date="2024-01-01", method="Tier 4")
    assert f"{t4['auc'] - t1['auc']:+.3f}" in section


def test_malacca_has_false_alarm_rates_and_no_auc() -> None:
    section = _section(rf.render_tables(SRC), "Malacca false-alarm rate")
    m = SRC["results_main"]
    m = m[m["region"] == "malacca"]
    assert m["test_positives"].max() == 0
    for method in ("Tier 1", "Tier 5", "M0 random"):
        r = _row(m, method=method)
        assert f"{r['far']:.3f}" in section, method
    assert "no positives" in section


def test_surrogate_table_matches_the_committed_summary() -> None:
    section = _section(rf.render_tables(SRC), "Surrogate fidelity")
    for region, rec in SRC["surrogate"]["regions"].items():
        assert f"{rec['r2_test']:.3f}" in section and f"{rec['r2_train']:.3f}" in section, region
        assert f"{rec['spearman_test']:.3f}" in section, region


def test_bem_table_matches_the_csv_and_names_each_feature() -> None:
    section = _section(rf.render_tables(SRC), "Bab el-Mandeb surrogate diagnostic")
    bem = SRC["bem"]
    r2 = bem[(bem["quantity"] == "r2") & (bem["feature"].fillna("") == "")]["value"].iloc[0]
    assert f"{r2:.3f}" in section
    for feature in bem["feature"].dropna().unique():
        if feature:
            assert feature in section


def test_lead_table_has_one_cell_per_onset_and_method() -> None:
    section = _section(rf.render_tables(SRC), "Lead time")
    lead = SRC["lead"]
    methods = list(dict.fromkeys(lead["method"]))
    for (region, onset), part in lead.groupby(["region", "onset"], sort=False):
        line = next(ln for ln in section.splitlines() if ln.startswith(f"| {region} | {onset} |"))
        cells = [c.strip() for c in line.strip().strip("|").split("|")][2:]
        expected = []
        for m in methods:
            r = part[part["method"] == m].iloc[0]
            expected.append("missed" if r["missed"] else
                            f"{r['lead_days']:+.0f}" + ("*" if r["censored_at_window"] else ""))
        assert cells == expected, (region, onset)
    assert "lagged 30-day mean" in section and "M0 random" in section


# ------------------------------------------------------------ provenance
@needs_git
def test_provenance_names_the_commit_and_the_inputs_by_blob_and_hash() -> None:
    prov = rf.provenance()
    assert re.fullmatch(r"[0-9a-f]{40}", prov["commit"])
    assert prov["commit"] == _git("rev-parse", "HEAD")        # the commit checked out now
    assert {i["path"] for i in prov["inputs"]} == set(rf.INPUTS)
    for item in prov["inputs"]:
        data = (_ROOT / item["path"]).read_bytes().decode("utf-8").replace("\r\n", "\n").encode("utf-8")
        assert item["sha256"] == hashlib.sha256(data).hexdigest(), item["path"]
        assert item["blob"] == hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(), item["path"]


@needs_git
def test_the_manifest_hash_and_its_blob_id_verification_are_computed() -> None:
    prov = rf.provenance()["manifest"]
    raw = (_ROOT / "outputs" / "MANIFEST.json").read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert prov["sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    manifest = json.loads(raw)
    assert prov["generated_utc"] == manifest["generated_utc"]
    assert prov["code_total"] == len(manifest["code"]["files"])
    expected_diff = sorted(e["path"] for e in manifest["code"]["files"]
                           if _git("rev-parse", f"HEAD:{e['path']}") != e["git_blob"])
    assert sorted(c["path"] for c in prov["code_mismatch"]) == expected_diff
    for c in prov["code_mismatch"]:                          # each recorded blob is in history
        assert c["last_commit_with_recorded"], c
    assert prov["committed_total"] == len(manifest["committed"])
    assert prov["committed_ok"] == prov["committed_total"]


@needs_git
def test_provenance_text_says_what_was_checked() -> None:
    prov = rf.provenance()
    text = rf.render_provenance(prov)
    assert "## Provenance" in text and "sha256" in text
    assert f"has sha256 `{prov['manifest']['sha256']}`" in text and prov["commit"] in text
    assert all(i["blob"] in text and i["sha256"] in text for i in prov["inputs"])
    for c in prov["manifest"]["code_mismatch"]:
        assert c["path"] in text and c["recorded"] in text
    doc = DOC.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert rf.provenance()["manifest"]["sha256"] in doc


# ---------------------------------------------- inputs must be committed
def _repo(tmp_path: Path) -> Path:
    def git(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path,
                       check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    git("add", "a.csv")
    git("commit", "-q", "-m", "x")
    return tmp_path


def test_inputs_must_be_committed_and_unchanged(tmp_path) -> None:
    repo = _repo(tmp_path)
    rf.check_inputs_committed(["a.csv"], repo)                         # clean: fine
    (repo / "a.csv").write_text("x\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a.csv"):
        rf.check_inputs_committed(["a.csv"], repo)                     # modified
    (repo / "b.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="b.csv"):
        rf.check_inputs_committed(["b.csv"], repo)                     # untracked


# ------------------------------------------------------- the command
@needs_git
def test_write_then_check_round_trip(tmp_path, monkeypatch) -> None:
    out = tmp_path / "RESULTS_FROZEN.md"
    monkeypatch.setattr(rf, "OUT", out)
    monkeypatch.setattr(rf, "check_inputs_committed", lambda *a, **k: None)
    assert rf.main(["--write"]) == 0 and out.exists()
    assert rf.main(["--check"]) == 0
    out.write_text(out.read_text(encoding="utf-8").replace("## Main results", "## Edited"),
                   encoding="utf-8")
    assert rf.main(["--check"]) == 1


def test_an_unknown_option_is_an_error(monkeypatch) -> None:
    monkeypatch.setattr(rf, "render", lambda *a, **k: pytest.fail("rendered"))
    with pytest.raises(SystemExit) as exc:
        rf.main(["--writ"])
    assert exc.value.code == 2


# ------------------------------------------- cells in the right columns (not just present)
def _cells(block: str, first: str) -> list[str]:
    line = next(ln for ln in block.splitlines() if ln.startswith(f"| {first} |"))
    return [c.strip() for c in line.strip().strip("|").split("|")]


def test_main_table_puts_auc_and_delta_in_their_own_columns() -> None:
    section = _section(rf.render_tables(SRC), "Main results")
    for region in ("hormuz", "bab_el_mandeb", "panama"):
        block = section.split(f"### {region}")[1].split("###")[0]
        r = _row(SRC["results_main"], region=region, method="Tier 5")
        cells = _cells(block, "Tier 5")
        assert cells[2].startswith(f"{r['auc']:.3f} [{r['auc_ci_low']:.3f}, {r['auc_ci_high']:.3f}]")
        assert f"{r['delta_auc']:+.3f} [{r['delta_auc_ci_low']:+.3f}, {r['delta_auc_ci_high']:+.3f}]" in cells[3]
        assert cells[4] == f"{r['alert_rate']:.3f}"
        assert _cells(block, "Tier 1")[3] == "—"                          # Tier 1 has no delta


def test_surrogate_columns_are_in_order() -> None:
    section = _section(rf.render_tables(SRC), "Surrogate fidelity")
    for region, rec in SRC["surrogate"]["regions"].items():
        cells = _cells(section, region)
        assert cells[3:6] == [f"{rec['r2_test']:.3f}", f"{rec['r2_train']:.3f}",
                              f"{rec['spearman_test']:.3f}"], region


def test_agent_ablation_puts_eq1_on_and_off_in_their_own_columns() -> None:
    section = _section(rf.render_tables(SRC), "Agent ablation")
    block = section.split("### panama")[1].split("###")[0]
    on = _row(SRC["ablation_on"], region="panama", method="shipping alone")
    off = _row(SRC["ablation_off"], region="panama", method="shipping alone")
    cells = _cells(block, "single")
    line = next(ln for ln in block.splitlines() if "shipping alone" in ln)
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    assert cells[3].startswith(f"{on['auc']:.3f}") and cells[4].startswith(f"{off['auc']:.3f}")
    assert cells[6] == f"{on['alert_rate']:.3f}"


def test_malacca_false_alarm_column_holds_the_false_alarm_rate() -> None:
    section = _section(rf.render_tables(SRC), "Malacca false-alarm rate")
    m = SRC["results_main"]
    m = m[m["region"] == "malacca"]
    for method in ("Tier 1", "B1 rolling z", "M0 random"):
        r = _row(m, method=method)
        cells = _cells(section, method)
        assert cells[2] == f"{r['far']:.3f}", method
        assert cells[3].startswith(f"{r['alert_rate']:.3f} [{r['alert_rate_ci_low']:.3f}"), method


def test_lead_cells_show_missed_and_censored_onsets() -> None:
    lead = pd.DataFrame({
        "region": ["hormuz"] * 4, "method": ["Tier 1", "M0 random"] * 2,
        "onset": ["2024-01-01"] * 2 + ["2024-06-01"] * 2,
        "lead_days": [12.0, 30.0, np.nan, -4.0],
        "censored_at_window": [False, True, False, False],
        "missed": [False, False, True, False], "alert_rate": [0.2, 0.1, 0.2, 0.1], "agents": ""})
    text = rf.table_lead(lead)
    first = _cells(text, "hormuz")
    assert first[1] == "2024-01-01" and first[2:] == ["+12", "+30*"]
    second = [ln for ln in text.splitlines() if "2024-06-01" in ln][0]
    assert "missed" in second and "-4" in second
    assert "2 of 4 method-onsets lead the label" in text.replace("**", "")


# --------------------------------------------- staged inputs are accepted, drifted ones not
def test_a_staged_input_is_accepted_and_one_changed_after_staging_is_not(tmp_path) -> None:
    repo = _repo(tmp_path)
    (repo / "new.csv").write_text("x\n9\n", encoding="utf-8")
    subprocess.run(["git", "add", "new.csv"], cwd=repo, check=True, capture_output=True)
    rf.check_inputs_committed(["a.csv", "new.csv"], repo)             # staged: part of this commit
    (repo / "new.csv").write_text("x\n10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="new.csv"):
        rf.check_inputs_committed(["new.csv"], repo)                  # drifted after staging


# ------------------------------------------------ review fixes: new behaviour
def test_leave_one_out_rows_carry_the_paired_change_with_its_interval() -> None:
    section = _section(rf.render_tables(SRC), "Agent ablation")
    block = section.split("### panama")[1].split("###")[0]
    r = _row(SRC["ablation_on"], region="panama", method="without shipping")
    line = next(ln for ln in block.splitlines() if "without shipping" in ln)
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    assert cells[5] == rf._paired(r, "loo_delta_auc") and cells[5] != "—"
    single = next(ln for ln in block.splitlines() if "shipping alone" in ln)
    assert [c.strip() for c in single.strip().strip("|").split("|")][5] == "—"


def test_a_method_that_alerts_every_day_is_flagged_and_a_blank_note_stays_blank() -> None:
    section = _section(rf.render_tables(SRC), "Agent ablation")
    block = section.split("### panama")[1].split("###")[0]
    always = SRC["ablation_on"]
    always = always[(always["region"] == "panama") & (always["alert_rate"] >= 1)]
    assert len(always) >= 1
    for method in always["method"]:
        line = next(ln for ln in block.splitlines() if f"| {method} |" in ln)
        assert "alerts on every scored day" in line
    assert "alerts on every scored day" not in next(
        ln for ln in block.splitlines() if "| full enabled set |" in ln)


def test_the_headline_split_is_marked_in_the_split_table() -> None:
    section = _section(rf.render_tables(SRC), "Split sensitivity")
    headline = str(SRC["results_main"]["test_start"].iloc[0])
    assert f"{headline} (headline split)" in section
    assert section.count("(headline split)") == len(SRC["split"]["region"].unique())


def test_the_page_explains_why_malacca_rates_coincide() -> None:
    section = _section(rf.render_tables(SRC), "Malacca false-alarm rate")
    assert "every alert is a false alarm" in section
    m = SRC["results_main"]
    m = m[m["region"] == "malacca"]
    assert (m["far"] - m["alert_rate"]).abs().max() < 1e-9      # so the sentence is true


def test_the_bem_note_explains_the_difference_in_test_days() -> None:
    section = _section(rf.render_tables(SRC), "Bab el-Mandeb surrogate diagnostic")
    bem = SRC["bem"]
    n = int(bem[(bem["quantity"] == "n") & (bem["feature"] == "")]["value"].iloc[0])
    scored = int(SRC["results_main"].query("region == 'bab_el_mandeb'")["n_test"].iloc[0])
    if n != scored:
        assert f"scores {n} test days" in section and f"score {scored} for the same region" in section
    else:
        assert "masked for missing news" not in section


def test_the_lead_section_states_the_window_and_the_chance_reference() -> None:
    section = _section(rf.render_tables(SRC), "Lead time")
    window = rf._lead_window()
    rates = SRC["lead"].drop_duplicates(["region", "method"])
    p = rates.loc[rates["method"] == "M0 random", "alert_rate"].mean()
    assert f"within {window} days either side" in section
    assert f"{p:.0%} of days, as the random control does" in section
    assert f"probability {1 - (1 - p) ** window:.0%}" in section
    assert f"The highest alert rate in this table is {rates['alert_rate'].max():.0%}" in section
    assert "upper reference" in section


@needs_git
def test_provenance_labels_a_new_input_and_says_how_to_verify_and_what_reruns_change() -> None:
    text = rf.render_provenance(rf.provenance())
    assert "git rev-parse <commit-or-tag>:<path>" in text
    assert "rerunning the script at the release tag" in text and "`--check` ignores them" in text
    assert "staged: the commit" not in text                       # the old jargon label
    prov = rf.provenance()
    for c in prov["manifest"]["code_mismatch"]:
        assert f"git diff {c['last_commit_with_recorded']} HEAD -- {c['path']}" in text
    assert f"{prov['manifest']['code_uncommitted_then']} of the producing files" in text


def test_committed_outputs_are_checked_against_the_blob_at_head(tmp_path) -> None:
    repo = _repo(tmp_path)
    data = (repo / "a.csv").read_bytes().replace(b"\r\n", b"\n")
    entry = {"path": "a.csv", "sha256": hashlib.sha256(data).hexdigest(), "git_blob": rf._blob(data)}
    assert rf.committed_outputs_ok([entry], repo) == 1
    (repo / "a.csv").write_text("x\n2\n", encoding="utf-8")            # worktree now differs from HEAD...
    changed = (repo / "a.csv").read_bytes().replace(b"\r\n", b"\n")
    drifted = {"path": "a.csv", "sha256": hashlib.sha256(changed).hexdigest(),
               "git_blob": rf._blob(changed)}                         # ...and matches a record of it
    assert rf.committed_outputs_ok([drifted], repo) == 0               # but HEAD does not hold it
    assert rf.committed_outputs_ok([{**entry, "path": "gone.csv"}], repo) == 0


def test_check_notices_a_stale_hash_not_only_stale_tables(tmp_path, monkeypatch, capsys) -> None:
    out = tmp_path / "RESULTS_FROZEN.md"
    monkeypatch.setattr(rf, "OUT", out)
    monkeypatch.setattr(rf, "check_inputs_committed", lambda *a, **k: None)
    prov = rf.provenance()
    page = rf.render()
    out.write_text(page, encoding="utf-8")
    assert rf.main(["--check"]) == 0
    out.write_text(page.replace(prov["inputs"][0]["blob"], "0" * 40), encoding="utf-8")
    assert rf.main(["--check"]) == 1 and "blob id or sha256 differs" in capsys.readouterr().out
    labelled = f"has sha256 `{prov['manifest']['sha256']}`"          # only the manifest sentence
    out.write_text(page.replace(labelled, "has sha256 `" + "0" * 64 + "`"), encoding="utf-8")
    assert rf.main(["--check"]) == 1 and "the manifest hash differs" in capsys.readouterr().out


def test_check_ignores_the_commit_cells(tmp_path, monkeypatch) -> None:
    out = tmp_path / "RESULTS_FROZEN.md"
    monkeypatch.setattr(rf, "OUT", out)
    monkeypatch.setattr(rf, "check_inputs_committed", lambda *a, **k: None)
    prov = rf.provenance()
    page = rf.render().replace(prov["commit"], "f" * 40).replace(rf._NEW, "abc1234")
    out.write_text(page, encoding="utf-8")
    assert rf.main(["--check"]) == 0


def test_a_drifted_input_and_a_missing_page_are_clean_messages_not_tracebacks(
        tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(rf, "OUT", tmp_path / "absent.md")
    monkeypatch.setattr(rf, "check_inputs_committed", lambda *a, **k: None)
    assert rf.main(["--check"]) == 1 and "does not exist" in capsys.readouterr().out

    def drifted(*a, **k):
        raise ValueError("inputs must be tracked and unchanged: x.csv (untracked)")

    monkeypatch.setattr(rf, "check_inputs_committed", drifted)
    assert rf.main(["--write"]) == 2 and "x.csv" in capsys.readouterr().out


def test_paired_cell_rules_on_synthetic_rows() -> None:
    def row(delta, lo, hi, valid=None):
        data = {"x_delta_auc": delta, "x_delta_auc_ci_low": lo, "x_delta_auc_ci_high": hi}
        if valid is not None:
            data["x_delta_valid"] = valid
        return pd.Series(data)

    assert rf._paired(row(-0.10, -0.15, -0.05), "x_delta_auc").startswith("**")      # clearly below 0
    assert rf._paired(row(0.10, 0.05, 0.15), "x_delta_auc").startswith("**")         # clearly above 0
    unclear = rf._paired(row(-0.02, -0.06, 0.03), "x_delta_auc")
    assert not unclear.startswith("**") and unclear == "-0.020 [-0.060, +0.030]"
    assert rf._paired(row(-0.1, -0.2, -0.05, valid=False), "x_delta_auc") == "—"     # invalid interval
    assert rf._paired(row(-0.1, -0.2, -0.05, valid=True), "x_delta_auc") != "—"
    assert rf._paired(row(np.nan, np.nan, np.nan), "x_delta_auc") == "—"
    near_zero = rf._paired(row(-0.0001, -0.0502, -0.0001), "x_delta_auc")
    assert "-0.0001" in near_zero and "-0.000 " not in near_zero                     # four decimals
