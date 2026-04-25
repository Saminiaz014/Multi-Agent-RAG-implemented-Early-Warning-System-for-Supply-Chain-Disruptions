"""Unit tests for the detection agent contracts."""

import numpy as np
import pandas as pd
import pytest

from src.agents.base_agent import BaseAgent, DetectionResult


class _ConcreteAgent(BaseAgent):
    """Minimal concrete agent that flags everything above 0.5 as anomalous."""

    def fit(self, df: pd.DataFrame) -> None:
        self._is_fitted = True

    def detect(self, df: pd.DataFrame) -> DetectionResult:
        scores = df.iloc[:, 0].to_numpy().astype(float)
        flags = scores > 0.5
        return DetectionResult(
            agent_name=self.name,
            anomaly_scores=scores,
            anomaly_flags=flags,
            feature_names=list(df.columns),
        )


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"signal": [0.1, 0.6, 0.9, 0.3, 0.8]})


@pytest.fixture()
def agent() -> _ConcreteAgent:
    return _ConcreteAgent(name="test_agent", config={})


def test_fit_sets_fitted_flag(agent: _ConcreteAgent, sample_df: pd.DataFrame) -> None:
    agent.fit(sample_df)
    assert agent._is_fitted is True


def test_detect_returns_detection_result(
    agent: _ConcreteAgent, sample_df: pd.DataFrame
) -> None:
    agent.fit(sample_df)
    result = agent.detect(sample_df)
    assert isinstance(result, DetectionResult)


def test_detection_result_shapes(
    agent: _ConcreteAgent, sample_df: pd.DataFrame
) -> None:
    result = agent.fit_detect(sample_df)
    assert result.anomaly_scores.shape == (len(sample_df),)
    assert result.anomaly_flags.shape == (len(sample_df),)


def test_detection_flags_correct_rows(
    agent: _ConcreteAgent, sample_df: pd.DataFrame
) -> None:
    result = agent.fit_detect(sample_df)
    expected_flags = np.array([False, True, True, False, True])
    np.testing.assert_array_equal(result.anomaly_flags, expected_flags)


def test_base_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseAgent(name="x", config={})  # type: ignore[abstract]
