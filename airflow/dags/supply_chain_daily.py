"""Daily orchestration of the supply-chain DSS pipeline (Option A, all local).

ingest_and_detect → rag_populate → evaluate → publish_marker, every day at
08:00 local time (AIRFLOW__CORE__DEFAULT_TIMEZONE). Hard-stop policy: no
retries, default all_success trigger rules — the first failing task fails the
run and nothing downstream executes, so the dashboard never sees a partial
refresh (the freshness marker is only written by the final task).

The DAG only orchestrates the project's existing entrypoints via
BashOperator; pipeline source is untouched.
"""

from __future__ import annotations

import logging
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

# Project root as mounted in docker-compose.yaml (bind mount of the host
# checkout — outputs written here land directly on the host disk).
PROJECT_ROOT = "/opt/airflow/supply-chain-dss"

# Quota-safe extractor subset for the DAILY RAG refresh — the single place to
# edit it. FRED (official limit 120 req/min) and Ambee tolerate a daily pull;
# NewsAPI / SerpAPI / ACLED free tiers would be exhausted at daily cadence,
# so they stay reserved for one-off manual backfills
# (python scripts/populate_knowledge_base.py with no --extractors flag).
DAILY_EXTRACTORS = "fred,ambee"


def _log_failure(context) -> None:
    """Local-log-only failure note (monitored via the Airflow UI, no email)."""
    ti = context.get("task_instance")
    logging.getLogger("supply_chain_daily").error(
        "hard-stop: task %r failed in run %s — downstream tasks will not run",
        getattr(ti, "task_id", "?"),
        context.get("run_id"),
    )


default_args = {
    "owner": "sami",
    "retries": 0,  # hard-stop: never retry, never partially refresh
    "on_failure_callback": _log_failure,
}

with DAG(
    dag_id="supply_chain_daily",
    description=(
        "Daily 6-agent pipeline → RAG populate (quota-safe subset) → "
        "evaluation → freshness marker for the local dashboard"
    ),
    schedule="0 8 * * *",
    start_date=datetime(2026, 7, 1),
    catchup=False,
    default_args=default_args,
    tags=["thesis", "supply-chain-dss"],
) as dag:
    # main.py is deliberately defensive: it catches pipeline exceptions,
    # prints a "PIPELINE FAILED: …" note and still exits 0. The grep below
    # promotes that note to a non-zero exit so Airflow's hard-stop holds
    # without modifying pipeline source. `pipefail` keeps a real python
    # crash (non-zero exit) fatal through the tee.
    ingest_and_detect = BashOperator(
        task_id="ingest_and_detect",
        bash_command=(
            "set -euo pipefail; "
            f"cd {PROJECT_ROOT}; "
            "python main.py 2>&1 | tee /tmp/ingest_and_detect.out; "
            "! grep -q 'PIPELINE FAILED' /tmp/ingest_and_detect.out"
        ),
    )

    rag_populate = BashOperator(
        task_id="rag_populate",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"python scripts/populate_knowledge_base.py --extractors {DAILY_EXTRACTORS}"
        ),
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=f"cd {PROJECT_ROOT} && python notebooks/evaluation.py",
    )

    # Freshness marker the dashboard reads (core.read_last_updated). Written
    # ONLY when every upstream task succeeded; ISO-8601 with local offset.
    publish_marker = BashOperator(
        task_id="publish_marker",
        bash_command=(
            f"cd {PROJECT_ROOT} && mkdir -p data/processed && "
            "date -Iseconds > data/processed/last_updated.txt && "
            "echo published: && cat data/processed/last_updated.txt"
        ),
    )

    ingest_and_detect >> rag_populate >> evaluate >> publish_marker
