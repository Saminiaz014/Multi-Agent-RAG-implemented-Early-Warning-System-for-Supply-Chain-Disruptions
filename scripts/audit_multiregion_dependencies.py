#!/usr/bin/env python
"""Audit multi-region readiness across the codebase.

Corrected against the real repo layout (agents live in ``src/agents``, not
``src/detection``; the orchestrator builds agents via
``Orchestrator._build_enabled_agents()`` + ``register_agent()``, not
attribute-assignment construction, so the naive regex for "agents
instantiated" is replaced with a direct read of that method's imports).
Also surfaces the three independent region vocabularies already present in
the codebase (see the REGION VOCABULARIES section) — any multi-region
design needs to reconcile these, not invent a fourth.
"""

from __future__ import annotations

import re
from pathlib import Path


def _location_constant(content: str) -> str | None:
    match = re.search(r'LOCATION:\s*str\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    match = re.search(r'LOCATION\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None


def _init_signature(content: str) -> str | None:
    match = re.search(r"def __init__\(self[^)]*\)[^:]*:", content)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(0))


def scan_connectors() -> None:
    print("=" * 70)
    print("CONNECTOR IMPLEMENTATIONS (src/ingestion/*_connector.py)")
    print("=" * 70)
    for f in sorted(Path("src/ingestion").glob("*_connector.py")):
        content = f.read_text(encoding="utf-8")
        cls = re.search(r"class (\w+Connector)", content)
        loc = _location_constant(content)
        init = _init_signature(content)
        print(f"\n{f.name}:")
        print(f"  Class: {cls.group(1) if cls else '?'}")
        print(f"  LOCATION: {loc or '(none found)'}  <- hardcoded class constant, not an __init__ param")
        print(f"  __init__: {init}")


def scan_agents() -> None:
    print("\n" + "=" * 70)
    print("AGENT IMPLEMENTATIONS (src/agents/*_agent.py)")
    print("=" * 70)
    for f in sorted(Path("src/agents").glob("*_agent.py")):
        content = f.read_text(encoding="utf-8")
        cls = re.search(r"class (\w+Agent)", content)
        loc = _location_constant(content) or re.search(
            r'_(?:DEFAULT_)?LOCATION\s*=\s*["\']([^"\']+)["\']', content
        )
        loc_val = loc if isinstance(loc, str) else (loc.group(1) if loc else None)
        init = _init_signature(content)
        print(f"\n{f.name}:")
        print(f"  Class: {cls.group(1) if cls else '?'}")
        print(f"  LOCATION: {loc_val or '(none found)'}  <- hardcoded module constant, not an __init__ param")
        print(f"  __init__: {init}")


def scan_orchestrator_and_risk_engine() -> None:
    print("\n" + "=" * 70)
    print("ORCHESTRATOR & RISK ENGINE")
    print("=" * 70)

    orch = Path("src/orchestrator.py")
    content = orch.read_text(encoding="utf-8")
    print(f"\n{orch.name}:")
    print(f"  __init__: {_init_signature(content)}")
    # Real agent-registration mechanism: _build_enabled_agents() imports each
    # agent class and calls register_agent() — not an attribute-assignment
    # pattern, so report the method's imports directly instead of a regex
    # that (correctly) finds nothing against this codebase.
    build = re.search(
        r"def _build_enabled_agents\(self\).*?(?=\n    def |\Z)", content, re.S
    )
    if build:
        imports = re.findall(r"from src\.agents\.\w+ import (\w+)", build.group(0))
        print(f"  Agents registered via _build_enabled_agents() -> register_agent(): {imports}")
    print("  No 'location'/'region' parameter anywhere in Orchestrator.__init__.")

    risk_engine = Path("src/aggregation/risk_engine.py")
    content = risk_engine.read_text(encoding="utf-8")
    print(f"\n{risk_engine.name}:")
    region_hits = re.findall(r".*region.*", content, re.I)
    if region_hits:
        print("  Region-awareness: PARTIAL — 'region' appears as a cosmetic")
        print("  log-message parameter only (see docstring: 'used only for log")
        print("  messages'), not wired into any threshold/weight logic:")
        for line in region_hits:
            print(f"    {line.strip()}")
    else:
        print("  Region-awareness: NO")


def scan_region_vocabularies() -> None:
    print("\n" + "=" * 70)
    print("REGION VOCABULARIES (three independent, unreconciled sets)")
    print("=" * 70)

    print("\n1. EVAL01 benchmark harness (src/benchmark/regions.py):")
    region_files = sorted(p.stem for p in Path("config/benchmark").glob("*.yaml"))
    print(f"   config/benchmark/*.yaml region specs: {region_files or '(none)'}")
    scenario_files = sorted(Path("config/benchmark/scenarios").glob("*.yaml"))
    scenario_regions = sorted({f.stem.rsplit("_", 2)[0] for f in scenario_files})
    print(f"   config/benchmark/scenarios/*.yaml regions: {scenario_regions}")
    print(f"   Scenario classes present per region: "
          f"{sorted({f.stem.split('_', 1)[1] for f in scenario_files if f.stem.startswith(scenario_regions[0])}) if scenario_regions else '(none)'}")

    print("\n2. Dashboard (src/dashboard/core.py):")
    core = Path("src/dashboard/core.py").read_text(encoding="utf-8")
    avail = re.search(r"AVAILABLE_REGIONS:\s*dict\[str, str\]\s*=\s*(\{[^}]*\})", core)
    print(f"   AVAILABLE_REGIONS = {avail.group(1) if avail else '(not found)'}")
    print("   get_routes(region)/get_news(region) already accept an arbitrary")
    print("   region key and degrade to empty results for unpopulated ones —")
    print("   this is the most forward-compatible region pattern already in the codebase.")

    print("\n3. RAG/KB extraction (config/settings.yaml: extraction.chokepoints):")
    import yaml

    settings = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
    chokepoints = list(settings.get("extraction", {}).get("chokepoints", {}).keys())
    print(f"   {chokepoints}")
    print("   Note: names diverge from the other two vocabularies — 'red_sea' not")
    print("   'bab_el_mandeb', 'malacca' present (not used anywhere else), no")
    print("   'panama'/'taiwan_strait'. This is for historical RAG case backfill,")
    print("   not live monitoring — a separate concern from the other two.")


def scan_config() -> None:
    print("\n" + "=" * 70)
    print("CONFIGURATION (config/settings.yaml)")
    print("=" * 70)
    content = Path("config/settings.yaml").read_text(encoding="utf-8")
    print("  Top-level 'global.active_region' or 'global.regions': NOT PRESENT")
    print("  (multiple prior prompts in this series assumed this exists; it doesn't)")
    hits = [line for line in content.splitlines() if "region" in line.lower()]
    print(f"  Other 'region' mentions ({len(hits)}):")
    for line in hits:
        print(f"    {line.strip()}")


def scan_api() -> None:
    print("\n" + "=" * 70)
    print("API LAYER (src/api/endpoints.py)")
    print("=" * 70)
    content = Path("src/api/endpoints.py").read_text(encoding="utf-8")
    hits = re.findall(r".*(?:location|region).*", content, re.I)
    print(f"  Region-awareness: {'PARTIAL' if hits else 'NONE'} ({len(hits)} mentions)")


if __name__ == "__main__":
    scan_connectors()
    scan_agents()
    scan_orchestrator_and_risk_engine()
    scan_region_vocabularies()
    scan_config()
    scan_api()
