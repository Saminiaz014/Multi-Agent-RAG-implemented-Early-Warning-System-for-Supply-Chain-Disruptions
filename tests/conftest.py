"""Suite-wide guards.

The weight optimizer writes two real artefacts:
``data/processed/optimization_results.json`` (read by
``notebooks/evaluation.py`` METRIC 5 and the API's
``/optimization/results``) and ``config/optimized_weights.yaml`` (read when
``weight_mode`` is ``"optimized"``). A test that runs the optimizer must
write to ``tmp_path``. A 5-trial smoke run in the suite kept overwriting the
real results (restored by a 100-trial rerun in 98213f2, then overwritten
again), so the run now fails if either file changes while it runs.
"""

from __future__ import annotations

import pytest

from tests._suite_guard import (REAL_OPTIMIZER_OUTPUTS, ROOT, describe_change,
                                fingerprint)


@pytest.fixture(scope="session", autouse=True)
def real_optimizer_outputs_unchanged():
    """Fail the run if a real optimizer output changed while the tests ran."""
    before = {path: fingerprint(path) for path in REAL_OPTIMIZER_OUTPUTS}
    yield
    changes = {str(path.relative_to(ROOT)): describe_change(before[path], fingerprint(path))
               for path in REAL_OPTIMIZER_OUTPUTS}
    changes = {path: how for path, how in changes.items() if how}
    assert not changes, (
        f"real optimizer outputs changed during the test run: {changes}. "
        "Tests must write optimizer results to tmp_path (or something else "
        "wrote these files while the suite ran)."
    )
