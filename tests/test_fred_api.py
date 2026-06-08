"""Standalone connectivity check for the FRED API.

This is intentionally not a pytest test — it is a single-file diagnostic
the user runs by hand to verify their FRED API key works against the
three series the market connector relies on
(:mod:`src.ingestion.market_connector`). It does not import or modify
any production code.

Usage
-----

1. Paste your real FRED key into ``.env`` at the project root::

       FRED_API_KEY=abcdef0123456789abcdef0123456789

   (Get a key at https://fred.stlouisfed.org/docs/api/api_key.html.)

2. Run::

       python tests/test_fred_api.py

3. On success, the script prints the last few observations for each
   series and the line ``All FRED endpoints verified. You can set
   source_mode: api in settings.yaml``.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - guarded import
    print(
        "ERROR: python-dotenv is not installed. Install with:\n"
        "    pip install python-dotenv --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(2)


FRED_BASE_URL: str = "https://api.stlouisfed.org/fred/series/observations"

# Series the market connector reads from FRED CSVs — testing the same
# IDs here confirms the API path will return the same data.
SERIES: tuple[tuple[str, str, str, int], ...] = (
    ("Brent crude (daily)", "DCOILBRENTEU", "DCOILBRENTEU", 5),
    ("Freight PPI (monthly)", "PCU4831114831115", "PCU4831114831115", 3),
    ("Freight services index (monthly)", "TSIFRGHTC", "TSIFRGHTC", 3),
)

REQUEST_TIMEOUT_SECONDS: int = 15


def _project_root() -> Path:
    """Return the project root (one level up from ``tests/``)."""
    return Path(__file__).resolve().parent.parent


def _load_api_key() -> str | None:
    """Load ``FRED_API_KEY`` from ``.env`` and return it.

    Returns:
        The API key string, or ``None`` if it is missing or still set to
        the placeholder ``"PASTE_YOUR_KEY_HERE"``.
    """
    env_path = _project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # falls back to CWD / process env

    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key or key == "PASTE_YOUR_KEY_HERE":
        return None
    return key


def _print_setup_instructions(env_path: Path) -> None:
    """Tell the user how to populate the FRED key."""
    print("=" * 70)
    print("FRED API key not configured.")
    print("=" * 70)
    print(f"  1. Open: {env_path}")
    print("  2. Replace 'PASTE_YOUR_KEY_HERE' with your real FRED API key.")
    print("     (Get one at https://fred.stlouisfed.org/docs/api/api_key.html)")
    print(f"  3. Re-run:  python {Path(__file__).relative_to(_project_root())}")
    print("=" * 70)


def _fetch_series(
    series_id: str, api_key: str, limit: int
) -> list[dict[str, str]]:
    """Hit the FRED observations endpoint and return the last ``limit`` rows.

    Args:
        series_id: FRED series identifier.
        api_key: FRED API key.
        limit: Number of trailing observations to request.

    Returns:
        List of ``{"date": ..., "value": ...}`` dicts ordered ascending.

    Raises:
        urllib.error.HTTPError / URLError: On HTTP failure.
        ValueError: If the JSON response is malformed or contains an error.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": str(limit),
    }
    url = f"{FRED_BASE_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        body = resp.read().decode("utf-8")
    payload = json.loads(body)
    if "error_code" in payload:
        raise ValueError(
            f"FRED API error {payload.get('error_code')}: "
            f"{payload.get('error_message')}"
        )
    observations = payload.get("observations", [])
    return list(reversed(observations))  # ascending date order for printing


def _check_one(name: str, series_id: str, limit: int, api_key: str) -> bool:
    """Print the last ``limit`` observations of a series; return success flag."""
    print(f"\n[{name}]  series_id={series_id}")
    try:
        rows = _fetch_series(series_id, api_key, limit)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"  FAIL: HTTP error — {exc}")
        return False
    except ValueError as exc:
        print(f"  FAIL: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  FAIL: {type(exc).__name__} — {exc}")
        return False

    if not rows:
        print("  FAIL: empty 'observations' array.")
        return False
    has_value = False
    for row in rows:
        date = row.get("date", "?")
        value = row.get("value", "?")
        print(f"  {date}  value={value}")
        if value not in (".", "?", ""):
            has_value = True
    if not has_value:
        print("  FAIL: all returned values are missing ('.').")
        return False
    print(f"  PASS: {len(rows)} observation(s) returned.")
    return True


def main() -> int:
    """Entry point. Returns a process exit code (0 = all checks passed)."""
    root = _project_root()
    env_path = root / ".env"
    print("=" * 70)
    print("FRED API connectivity check")
    print("=" * 70)
    print(f"  project root : {root}")
    print(f"  .env path    : {env_path}  exists={env_path.exists()}")

    key = _load_api_key()
    if key is None:
        _print_setup_instructions(env_path)
        return 1
    print(f"  key prefix   : {key[:4]}...{key[-4:]}  (len={len(key)})")

    results: list[tuple[str, bool]] = []
    for name, series_id, _label, limit in SERIES:
        ok = _check_one(name, series_id, limit, key)
        results.append((name, ok))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL':<5}  {name}")
    print("=" * 70)

    if all(ok for _, ok in results):
        print(
            "All FRED endpoints verified. You can set source_mode: api in "
            "settings.yaml"
        )
        return 0
    print("One or more endpoints failed — see messages above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
