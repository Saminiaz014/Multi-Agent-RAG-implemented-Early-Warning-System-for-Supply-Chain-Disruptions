"""Weight-optimization layer.

Houses the Optuna-driven weight tuner, the train/val/test split manager,
the active-weight resolver that backs the ``weight_mode`` switch, and the
post-hoc optimization-analysis/visualisation helpers.
"""

from src.optimization.weight_config import (
    apply_weights_to_agent,
    load_optimized_weights,
    optimized_weights_path,
    resolve_active_weights,
)

__all__ = [
    "apply_weights_to_agent",
    "load_optimized_weights",
    "optimized_weights_path",
    "resolve_active_weights",
]
