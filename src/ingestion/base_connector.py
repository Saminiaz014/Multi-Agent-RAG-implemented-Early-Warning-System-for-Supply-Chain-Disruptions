"""Abstract base class for all data source connectors."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseConnector(ABC):
    """Contract that every data connector must satisfy.

    A connector is responsible for fetching raw signal data from a single
    source (e.g., AIS vessel tracking, oil futures, news feeds) and
    returning it as a normalised pandas DataFrame.

    Args:
        config: Domain-specific configuration block from settings.yaml.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Retrieve raw data from the source.

        Returns:
            Raw signal data with at minimum a ``timestamp`` column and
            one or more numeric feature columns.
        """

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """Check that the fetched DataFrame meets schema expectations.

        Args:
            df: DataFrame returned by :meth:`fetch`.

        Returns:
            True if the data passes all quality checks, False otherwise.
        """

    def fetch_and_validate(self) -> pd.DataFrame:
        """Fetch data and raise if validation fails.

        Returns:
            Validated DataFrame ready for the ingestion pipeline.

        Raises:
            ValueError: If :meth:`validate` returns False.
        """
        df = self.fetch()
        if not self.validate(df):
            raise ValueError(
                f"{self.__class__.__name__}: fetched data failed validation."
            )
        return df
