"""Suite-wide guards.

The suite fails if a real output changes while it runs: the optimizer's two
artefacts below, and the two SHAP comparison plots
(``data/processed/shap_comparison_{waterfall,importance}.png``, shown by the
dashboard; ``test_comparison_plots_saved`` used to overwrite them on every run).

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

from tests._suite_guard import GUARDED_OUTPUTS, ROOT, describe_change, fingerprint


@pytest.fixture(scope="session", autouse=True)
def real_optimizer_outputs_unchanged():
    """Fail the run if a real output (optimizer results, SHAP plots) changed while the tests ran."""
    before = {path: fingerprint(path) for path in GUARDED_OUTPUTS}
    yield
    changes = {str(path.relative_to(ROOT)): describe_change(before[path], fingerprint(path))
               for path in GUARDED_OUTPUTS}
    changes = {path: how for path, how in changes.items() if how}
    assert not changes, (
        f"real outputs changed during the test run: {changes}. "
        "Tests must write optimizer results and SHAP plots to tmp_path (or something "
        "else wrote these files while the suite ran)."
    )
