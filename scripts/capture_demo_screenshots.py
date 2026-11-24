"""Capture the Chapter 3 demonstration screenshots from the running dashboard.

For each case in ``scripts/case_metrics.py`` this opens the dashboard by deep
link (no clicking through menus), waits for Streamlit to finish rendering and
saves element screenshots to ``thesis/figures/ch3_demo/``:

* ``decision_<case>_<date>.png`` — Decision view on each snapshot date (map,
  status, recommended action and the peak explanation);
* ``analysis_risk_<case>.png`` / ``analysis_agents_<case>.png`` — the real-data
  explorer's composite and per-agent charts over the case window;
* ``analysis_shap_<case>_<date>.png`` — SHAP drivers on the main snapshot day;
* ``analysis_news_panama.png`` — the news-sentiment panel (Case 4).

Snapshot dates and corridors come from ``case_metrics.json`` (run
``scripts/case_metrics.py`` first), so figures and text use the same days.
Starts Streamlit on the port if nothing is serving there.

Usage::

    python scripts/capture_demo_screenshots.py [--port 8501]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _PROJECT_ROOT / "thesis" / "figures" / "ch3_demo"
MIN_BYTES = 20_000


def metrics_sha256(path: Path) -> str:
    """Hex sha256 of ``case_metrics.json``: the manifest ties the pictures to the numbers."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stale_captures(manifest: dict, current: str) -> list[str]:
    """Captures taken against other numbers than ``current``, or against none on record.

    A run with ``--only`` keeps the other page type's earlier captures, so the manifest's own
    hash can vouch for pictures it did not take; each capture therefore carries its own.
    """
    return [c["file"] for c in manifest.get("captures", [])
            if c.get("case_metrics_sha256") != current]


def serving(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/_stcore/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def start_streamlit(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py",
         "--server.headless", "true", "--server.port", str(port),
         "--server.address", "localhost"],
        cwd=_PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://localhost:{port}"
    for _ in range(120):
        if serving(base):
            return proc
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("Streamlit did not come up")


def settle(page, selector: str, timeout_s: int = 600) -> None:
    """Wait for ``selector``, then for Streamlit to stop running for 3 s."""
    page.wait_for_selector(selector, state="visible", timeout=timeout_s * 1000)
    quiet_since = None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        running = page.locator('[data-testid="stStatusWidget"]').count() > 0
        if running:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
        elif time.time() - quiet_since >= 3:
            break
        time.sleep(0.5)
    page.wait_for_timeout(2500)   # plotly transitions and map tiles


def check_png(path: Path) -> tuple[bool, str]:
    from PIL import Image, ImageStat

    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size <= MIN_BYTES:
        return False, f"{size} bytes"
    stat = ImageStat.Stat(Image.open(path).convert("L"))
    if stat.stddev[0] < 2.0:
        return False, "uniform colour"
    return True, f"{size // 1024} KB"


def capture_decision(browser, base: str, case: dict, snap: dict, path: Path) -> None:
    dv = snap["decision_view"]
    query = urlencode({"region": case["region"], "route": dv["route_key"],
                       "date": snap["date"], "window": "7"})
    # Streamlit scrolls an inner container, not the window, so a full-page
    # capture only ever sees one viewport. A viewport tall enough for the map,
    # the decision column and the explanation avoids scrolling altogether.
    page = browser.new_page(viewport={"width": 1600, "height": 2400}, device_scale_factor=2)
    page.goto(f"{base}/?{query}", wait_until="domcontentloaded")
    settle(page, '[data-testid="stPlotlyChart"]')
    page.get_by_role("button", name="Explain the marked peak").click()
    settle(page, ".explain-box")
    page.wait_for_timeout(4000)   # MapLibre tiles inside the iframe
    bottom = page.evaluate("""() => {
        document.querySelectorAll('*').forEach(el => { if (el.scrollTop) el.scrollTop = 0; });
        const exp = document.querySelector('.explain-box').getBoundingClientRect();
        const box = exp.bottom;
        let right = 0;   // the decision column beside the map, not lower panels
        document.querySelectorAll('[data-testid="stColumn"]').forEach(c => {
            const r = c.getBoundingClientRect();
            if (r.left > 900 && r.top < exp.top) right = Math.max(right, r.bottom);
        });
        return Math.min(Math.max(box + 70, right + 10), 2400);
    }""")
    page.wait_for_timeout(1500)
    page.screenshot(path=str(path), clip={"x": 0, "y": 0, "width": 1600, "height": bottom})
    page.close()


def capture_analysis(browser, base: str, case: dict, date: str, shots: dict) -> None:
    query = urlencode({"view": "cases", "region": case["region"], "start": case["window"][0],
                       "end": case["window"][1], "date": date})
    page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
    page.goto(f"{base}/Analysis_View?{query}", wait_until="domcontentloaded")
    settle(page, ".st-key-case_shap [data-testid='stPlotlyChart']")
    for key, path in shots.items():
        element = page.locator(f".st-key-{key}")
        element.scroll_into_view_if_needed()
        page.wait_for_timeout(800)
        element.screenshot(path=str(path))
    page.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--only", choices=("decision", "analysis"),
                        help="Capture one page type only (the manifest keeps the other's entries).")
    args = parser.parse_args(argv)
    metrics = json.loads((OUT_DIR / "case_metrics.json").read_text(encoding="utf-8"))
    base = f"http://localhost:{args.port}"
    proc = None if serving(base) else start_streamlit(args.port)

    from playwright.sync_api import sync_playwright

    jobs: list[tuple[dict, str, callable, Path]] = []
    for case in metrics["cases"]:
        slug = f"case{case['case']}"
        for snap in case["snapshots"]:
            path = OUT_DIR / f"decision_{slug}_{snap['date']}.png"
            jobs.append(({"case": case["case"], "region": case["region"], "page": "decision",
                          "date": snap["date"], "route": snap["decision_view"]["route_key"],
                          "file": path.name},
                         "decision", (case, snap), path))
        main_date = case["snapshots"][0]["date"]
        shots = {"case_risk": OUT_DIR / f"analysis_risk_{slug}.png",
                 "case_agents": OUT_DIR / f"analysis_agents_{slug}.png",
                 "case_shap": OUT_DIR / f"analysis_shap_{slug}_{main_date}.png"}
        if case["region"] == "panama":
            shots["case_news"] = OUT_DIR / "analysis_news_panama.png"
        jobs.append(({"case": case["case"], "region": case["region"], "page": "analysis",
                      "window": case["window"], "date": main_date,
                      "files": [p.name for p in shots.values()]},
                     "analysis", (case, main_date, shots), None))

    manifest, results = [], []
    metrics_hash = metrics_sha256(OUT_DIR / "case_metrics.json")
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--use-angle=swiftshader",
                                               "--enable-unsafe-swiftshader"])
            for meta, kind, payload, path in jobs:
                if args.only and kind != args.only:
                    continue
                files = [path] if path else list(payload[2].values())
                for attempt in (1, 2):
                    try:
                        if kind == "decision":
                            capture_decision(browser, base, *payload, path)
                        else:
                            capture_analysis(browser, base, *payload)
                    except Exception as exc:
                        print(f"  attempt {attempt} failed: {exc}")
                    checks = [check_png(f) for f in files]
                    if all(ok for ok, _ in checks):
                        break
                for f, (ok, note) in zip(files, checks):
                    results.append((f.name, ok, note))
                    print(f"{'PASS' if ok else 'FAIL'}  {f.name}  ({note})")
                manifest.append({**meta, "case_metrics_sha256": metrics_hash,
                                 "captured": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"), "ok": all(ok for ok, _ in checks)})
            browser.close()
    finally:
        if proc is not None:
            proc.terminate()

    previous = OUT_DIR / "manifest.json"
    if args.only and previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        names = {n for n, _, _ in results}
        manifest = [m for m in old["captures"] if m["page"] != args.only] + manifest
        results = [(f["file"], f["ok"], f["note"]) for f in old["files"]
                   if f["file"] not in names] + results
    (OUT_DIR / "manifest.json").write_text(json.dumps(
        {"base_url": base, "viewport": [1600, 1000], "device_scale_factor": 2,
         "case_metrics_sha256": metrics_sha256(OUT_DIR / "case_metrics.json"),
         "captures": manifest, "files": [{"file": n, "ok": ok, "note": note}
                                          for n, ok, note in results]}, indent=2),
        encoding="utf-8")
    failed = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} screenshots passed")
    stale = stale_captures({"captures": manifest}, metrics_hash)
    if stale:
        print(f"WARNING: {len(stale)} capture(s) were taken against other numbers than "
              f"{OUT_DIR / 'case_metrics.json'}: {', '.join(stale)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
