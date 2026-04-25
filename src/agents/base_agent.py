"""Abstract base class for all anomaly detection agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DetectionResult:
    """Output produced by a single detection agent.

    Attributes:
        agent_name: Identifier of the agent that produced this result.
        anomaly_scores: Per-sample anomaly score in [0, 1].
        anomaly_flags: Boolean mask — True where an anomaly is detected.
        feature_names: Names of the features used during detection.
        metadata: Arbitrary extra info (model params, timestamps, etc.).
    """

    agent_name: str
    anomaly_scores: np.ndarray
    anomaly_flags: np.ndarray
    feature_names: list[str]
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Contract that every detection agent must satisfy.

    Each agent wraps a single anomaly detection algorithm and operates on
    a feature-ready DataFrame. Agents are domain-scoped (shipping, market,
    geopolitical) and run independently before cross-agent aggregation.

    Args:
        name: Human-readable agent identifier.
        config: Agent-specific configuration block from settings.yaml.
    """

    def __init__(self, name: str, config: dict) -> None:
        self.name = name
        self.config = config
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train or calibrate the underlying detection model.

        Args:
            df: Historical feature DataFrame used for fitting.
        """

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> DetectionResult:
        """Run anomaly detection on new observations.

        Args:
            df: Feature DataFrame with the same schema used in :meth:`fit`.

        Returns:
            :class:`DetectionResult` with scores and flags for each row.
        """

    def fit_detect(self, df: pd.DataFrame) -> DetectionResult:
        """Convenience method: fit then detect on the same data.

        Useful for one-shot evaluation or when training and inference data
        are the same (e.g., unsupervised retrospective analysis).

        Args:
            df: Feature DataFrame.

        Returns:
            :class:`DetectionResult` from :meth:`detect`.
        """
        self.fit(df)
        return self.detect(df)
