"""
Build daily news CSVs for the 4 chokepoints from GDELT 1.0 daily event files.

- Static file download (data.gdeltproject.org), so there is no API, no key and no rate limit.
- Each daily zip is streamed, filtered to the corridor bounding boxes, aggregated, then discarded.
- Output: data/eval/news/<region>.csv  +  <region>.manifest.json (sha256, params)
- Resumable: days already processed are skipped (progress kept in data/eval/news/_progress.json).
- A day with no corridor events is a real observation (0 volume, "peace time"), not missing data.
  Only days whose export file does not exist are missing; they get no row.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/build_gdelt_news_csv.py --start 2019-01-01 --end 2026-11-20
    .venv/Scripts/python.exe scripts/build_gdelt_news_csv.py --check        # 1-day sanity check only
    .venv/Scripts/python.exe scripts/build_gdelt_news_csv.py --zero-check   # 2019-01-01..03-31 zero-day share

FILTER CHANGELOG (a change to BBOXES, KEEP_GEO_TYPES or EXCLUDE_NAMES needs a new
FILTER_VERSION and a logged reason here):
  v1 2026-11-21  Water-only boxes from map/gazetteer coordinates, before any result was seen.
                 Geo types 3/4 only; Panama City excluded by name.
  v2 2026-11-21  Malacca and Panama boxes enlarged at the author's decision. Reason: on the
                 2019-01-01..2019-03-31 sample under v1, Malacca had 100% zero-volume days
                 and Panama 58.9% (Hormuz 47.8%, Bab el-Mandeb 83.3%). The new boxes were
                 sized to contain the coordinates GDELT itself assigns to the corridor's
                 named features (see GAZETTEER NOTE below). Hormuz and Bab el-Mandeb unchanged.
"""
import argparse, csv, datetime as dt, hashlib, http.client, io, json, os, sys, time, zipfile, zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

FILTER_VERSION = "v2"
FROZEN_AT = "2026-11-21"

# ---- Corridor boxes: region -> list of (lat_min, lat_max, lon_min, lon_max) ----
# An event counts for a region if its ActionGeo point lies in ANY of its boxes.
#
# GAZETTEER NOTE (v2): GDELT's own ActionGeo coordinates for the corridor features,
# read from the 2019-01-15, 2023-08-15, 2024-01-15 and 2024-06-01 export files:
#   "Panama Canal, Panama (general), Panama"  type 4  9.33333, -79.9167  (outside v1: lat > 9.32)
#   "Panama City, Panama, Panama"             type 4  8.96667, -79.5333  (excluded by name)
#   "Malacca, Melaka, Malaysia" / "Melaka"    type 4  2.196,   102.241   (outside v1)
#   no separate "Strait of Malacca" point appeared; strait coverage geocodes to Malacca/Melaka.
#
# hormuz (26.0-26.9 N, 55.9-57.2 E)  [unchanged from v1]
#   inside:  Khasab, Bukha, Kumzar, Larak Is., Kuhestak (small ports/villages)
#   outside: Qeshm town, Hormuz Is., Bandar Abbas, Hengam Is., Ras al-Khaimah, Dibba,
#            Jask, Minab, Dubai
# bab_el_mandeb (12.2-13.0 N, 43.0-43.8 E)  [unchanged from v1]
#   inside:  Perim/Mayyun Is., Dhubab, Ras Siyyan, Khor Angar, Moulhoule, Rahaita
#   outside: Mocha, Assab, Obock, Tadjoura, Djibouti city, Aden, Little Aden
# panama (8.85-9.40 N, 80.05-79.52 W)  [v2: enlarged from 8.90-9.32 N, 79.98-79.53 W]
#   inside:  GDELT "Panama Canal" point, Colon (9.36, -79.90), Cristobal, Gatun Lake and
#            locks, Gamboa, Pedro Miguel/Miraflores locks, Balboa, Arraijan, Howard,
#            La Chorrera (8.88, -79.78); Panama City point inside but excluded by name
#   outside: San Miguelito (9.03, -79.50), Tocumen (9.08, -79.38), Portobelo (9.55, -79.65)
# malacca  [v2: v1 open-water box kept, southern-strait box added]
#   box 1 (3.3-4.8 N, 99.6-100.5 E): open water, no towns
#   box 2 (1.6-2.6 N, 100.8-102.6 E): southern strait, contains GDELT's Malacca/Melaka point
#     inside:  Melaka city (2.196, 102.241), Port Dickson (2.52, 101.80), Muar (2.04, 102.57),
#              Dumai (1.67, 101.45), Rupat Is., Alor Gajah (2.38, 102.21)
#     outside: Bengkalis (1.47, 102.10), Seremban (2.72, 101.94), Kuala Lumpur (3.14, 101.69), Port Klang (3.00, 101.39),
#              Batu Pahat (1.85, 102.93), Singapore (1.35, 103.82)
BBOXES = {
    "hormuz":        [(26.0, 26.9,  55.9,  57.2)],
    "bab_el_mandeb": [(12.2, 13.0,  43.0,  43.8)],
    "panama":        [( 8.85, 9.40, -80.05, -79.52)],
    "malacca":       [( 3.3,  4.8,  99.6, 100.5), (1.6, 2.6, 100.8, 102.6)],
}
# GDELT geo types: 1 country, 2 US state, 3 US city, 4 world city/feature, 5 world ADM1.
# Only point geocodes (3, 4) are kept; country and ADM1 centroids are not corridor-specific.
KEEP_GEO_TYPES = frozenset({"3", "4"})
# Lower-cased first component of ActionGeo_FullName, excluded per region.
EXCLUDE_NAMES = {"panama": frozenset({"panama city"})}

#: Content hash of the filter, so the version claim is checkable from any manifest.
FILTER_SHA256 = hashlib.sha256(json.dumps(
    {"bboxes": BBOXES, "geo_types": sorted(KEEP_GEO_TYPES),
     "exclude_names": {k: sorted(v) for k, v in EXCLUDE_NAMES.items()}},
    sort_keys=True).encode()).hexdigest()

URL = "https://data.gdeltproject.org/events/{d}.export.CSV.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "news"
#: A 404 younger than this is "not published yet" and is retried next run.
PENDING_GRACE_DAYS = 3
PENDING = object()

# GDELT 1.0 daily event file: 58 tab-separated columns (0-based indices).
# Each row is dated by its EXPORT FILE (the day GDELT added the event, = DATEADDED),
# not by SQLDATE: day t only uses news published by day t (no lookahead).
C_DATE, C_QUAD, C_GOLD, C_NART, C_TONE = 1, 29, 30, 33, 34
C_AGEO_TYPE, C_AGEO_NAME, C_AGEO_LAT, C_AGEO_LON = 49, 50, 53, 54
N_COLS = 58

_RETRYABLE = (URLError, http.client.HTTPException, ConnectionError, TimeoutError,
              OSError, zipfile.BadZipFile, zlib.error, EOFError)


def _valid_zip(blob: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            return bool(z.namelist()) and z.testzip() is None
    except (zipfile.BadZipFile, zlib.error, EOFError):
        return False


def fetch(day: dt.date, tries: int = 5, today: dt.date | None = None):
    """Download one daily zip.

    Returns the bytes; ``None`` for a 404 older than PENDING_GRACE_DAYS (a
    genuinely missing day); ``PENDING`` for a 404 on a recent day (not yet
    published, retried next run). Truncated/corrupt downloads are retried.
    """
    today = today or dt.date.today()
    url = URL.format(d=day.strftime("%Y%m%d"))
    for i in range(tries):
        try:
            with urlopen(Request(url, headers={"User-Agent": "thesis-research"}), timeout=120) as r:
                blob = r.read()
            if _valid_zip(blob):
                return blob
        except HTTPError as e:
            if e.code == 404:
                return PENDING if (today - day).days <= PENDING_GRACE_DAYS else None
        except _RETRYABLE:
            pass
        if i < tries - 1:
            time.sleep(5 * 2 ** i)
    raise RuntimeError(f"failed after {tries} tries: {url}")


def region_of(lat: float, lon: float) -> str | None:
    for name, boxes in BBOXES.items():
        for a, b, c, d in boxes:
            if a <= lat <= b and c <= lon <= d:
                return name
    return None


def empty_agg() -> dict:
    return dict(n_events=0, n_articles=0, tone_x_art=0.0, goldstein_sum=0.0, n_material_conflict=0)


def process(blob: bytes) -> dict:
    """Return {region: aggregates} for one day."""
    agg = defaultdict(empty_agg)
    bad_rows = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                p = line.rstrip("\r\n").split("\t")
                if len(p) != N_COLS:
                    bad_rows += 1
                    continue
                if p[C_AGEO_TYPE] not in KEEP_GEO_TYPES or not p[C_AGEO_LAT]:
                    continue
                try:
                    lat, lon = float(p[C_AGEO_LAT]), float(p[C_AGEO_LON])
                except ValueError:
                    bad_rows += 1
                    continue
                reg = region_of(lat, lon)
                if reg is None:
                    continue
                name = p[C_AGEO_NAME].split(",")[0].strip().lower()
                if name in EXCLUDE_NAMES.get(reg, ()):
                    continue
                try:
                    n_art = int(p[C_NART] or 0)
                    tone, gold = float(p[C_TONE] or 0), float(p[C_GOLD] or 0)
                except ValueError:
                    bad_rows += 1
                    continue
                a = agg[reg]
                a["n_events"] += 1
                a["n_articles"] += n_art
                a["tone_x_art"] += tone * n_art
                a["goldstein_sum"] += gold
                a["n_material_conflict"] += p[C_QUAD] == "4"
    return {"agg": dict(agg), "bad_rows": bad_rows}


def row(day, a):
    n_e, n_a = a["n_events"], a["n_articles"]
    return {
        "date": day.isoformat(),
        "news_volume": n_a,                                   # articles about events in corridor
        "n_events": n_e,
        "news_tone": round(a["tone_x_art"] / n_a, 4) if n_a else "",   # NaN when no coverage
        "goldstein_mean": round(a["goldstein_sum"] / n_e, 4) if n_e else "",
        "n_material_conflict": a["n_material_conflict"],
    }


def missing_in_range(missing: list[str], start: dt.date, end: dt.date) -> list[str]:
    return sorted(k for k in missing if start.isoformat() <= k <= end.isoformat())


def _days(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def _load_progress(prog_f: Path) -> dict:
    if not prog_f.exists():
        return {"filter_version": FILTER_VERSION, "filter_sha256": FILTER_SHA256,
                "done": {}, "missing": []}
    prog = json.loads(prog_f.read_text())
    if (prog.get("filter_version"), prog.get("filter_sha256")) != (FILTER_VERSION, FILTER_SHA256):
        raise SystemExit(f"{prog_f} was built with filter {prog.get('filter_version')!r} "
                         f"({str(prog.get('filter_sha256'))[:12]}), script is {FILTER_VERSION!r} "
                         f"({FILTER_SHA256[:12]}): move it aside before rebuilding.")
    return prog


def _save_progress(prog: dict, prog_f: Path) -> None:
    """Atomic write: a crash mid-write must not destroy the resume state."""
    tmp = prog_f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(prog))
    os.replace(tmp, prog_f)


def _fetch_process(day: dt.date):
    """Worker: fetch and aggregate one day. Returns (kind, payload)."""
    try:
        blob = fetch(day)
        if blob is PENDING:
            return "pending", None
        if blob is None:
            return "missing", None
        res = process(blob)
    except Exception as exc:                  # one bad day must never stop the backfill
        return "failed", f"{type(exc).__name__}: {str(exc)[:120]}"
    time.sleep(1)                             # polite to the file server
    return "ok", {"agg": res["agg"], "bad_rows": res["bad_rows"]}


def _ingest(prog: dict, prog_f: Path, start: dt.date, end: dt.date,
            workers: int = 1) -> tuple[list[str], list[str]]:
    """Fetch every unprocessed day in range.

    Returns ``(pending, failed)``. Neither is persisted, so a rerun retries both;
    a failed day never stops the run.
    """
    todo = [d for d in _days(start, end)
            if d.isoformat() not in prog["done"] and d.isoformat() not in prog["missing"]]
    pending: list[str] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for n, (day, (kind, payload)) in enumerate(zip(todo, pool.map(_fetch_process, todo)), 1):
            k = day.isoformat()
            if kind == "pending":
                pending.append(k)
            elif kind == "failed":
                failed.append(k)
            else:
                if kind == "missing":
                    prog["missing"].append(k)
                else:
                    prog["done"][k] = payload
                _save_progress(prog, prog_f)
            print(f"{k} {kind} ({n}/{len(todo)})", flush=True)
    return pending, failed


def _zero_share(prog: dict, start: dt.date, end: dt.date) -> dict[str, tuple[int, int]]:
    done = [prog["done"][d.isoformat()] for d in _days(start, end) if d.isoformat() in prog["done"]]
    return {reg: (sum(1 for x in done if x["agg"].get(reg, empty_agg())["n_articles"] == 0), len(done))
            for reg in BBOXES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=(dt.date.today() - dt.timedelta(days=1)).isoformat())
    ap.add_argument("--workers", type=int, default=4, help="concurrent downloads")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--zero-check", action="store_true",
                    help="ingest 2019-01-01..2019-03-31 and report the zero-volume share per region")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.check:
        day = dt.date(2024, 1, 15)
        blob = fetch(day)
        if blob is None or blob is PENDING:
            raise SystemExit(f"{day}: export file not available")
        res = process(blob)
        print(day, "bad_rows:", res["bad_rows"])
        for reg in BBOXES:
            print(reg, row(day, res["agg"].get(reg, empty_agg())))
        return 0

    prog_f = OUT / "_progress.json"
    prog = _load_progress(prog_f)

    if args.zero_check:
        start, end = dt.date(2019, 1, 1), dt.date(2019, 3, 31)
        _ingest(prog, prog_f, start, end, args.workers)
        for reg, (zeros, n) in _zero_share(prog, start, end).items():
            print(f"{reg:<14} zero-volume days {zeros}/{n} = {zeros / n if n else 1:.1%}")
        print("missing days:", missing_in_range(prog["missing"], start, end))
        return 0

    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    if start > end:
        raise SystemExit("--start is after --end")
    pending, failed = _ingest(prog, prog_f, start, end, args.workers)
    if failed:                                # one more pass, sequential, before giving up
        print(f"retrying {len(failed)} failed days", flush=True)
        retry_start = dt.date.fromisoformat(min(failed))
        retry_end = dt.date.fromisoformat(max(failed))
        p2, failed = _ingest(prog, prog_f, retry_start, retry_end, 1)
        pending = sorted(set(pending) | set(p2))

    # One CSV per region. Every processed day gets a row (0 volume, empty tone if
    # no events). Missing, pending and failed days get NO row: they are listed in
    # the manifest, never filled.
    n_cal = (end - start).days + 1
    missing = missing_in_range(prog["missing"], start, end)
    zero = _zero_share(prog, start, end)
    for reg in BBOXES:
        path = OUT / f"{reg}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row(start, empty_agg()).keys()))
            w.writeheader()
            for d in _days(start, end):
                k = d.isoformat()
                if k in prog["done"]:
                    w.writerow(row(d, prog["done"][k]["agg"].get(reg, empty_agg())))
        n_rows = sum(1 for _ in path.open()) - 1
        zeros, n_done = zero[reg]
        manifest = {
            "source": "GDELT 1.0 daily event export (data.gdeltproject.org/events)",
            "filter_version": FILTER_VERSION, "filter_sha256": FILTER_SHA256,
            "frozen_at": FROZEN_AT, "day_basis": "export file date (DATEADDED), not SQLDATE",
            "region": reg, "bboxes_lat_lat_lon_lon": BBOXES[reg],
            "filter": "ActionGeo inside any bbox, ActionGeo_Type in {3,4}"
                      + (f", ActionGeo_FullName not in {sorted(EXCLUDE_NAMES[reg])}"
                         if reg in EXCLUDE_NAMES else ""),
            "start": args.start, "end": args.end,
            "calendar_days": n_cal, "rows": n_rows,
            "missing_days": missing, "pending_days": pending, "failed_days": failed,
            # share of calendar days with no row (missing + pending + failed)
            "missing_pct": round(100 * (n_cal - n_rows) / n_cal, 3),
            # zero-volume days are real observations (no corridor events), not missing
            "zero_volume_days": zeros,
            "zero_volume_pct": round(100 * zeros / n_done, 3) if n_done else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        (OUT / f"{reg}.manifest.json").write_text(json.dumps(manifest, indent=2))
        print(reg, n_rows, "rows", f"zero-volume {manifest['zero_volume_pct']}%",
              manifest["sha256"][:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
