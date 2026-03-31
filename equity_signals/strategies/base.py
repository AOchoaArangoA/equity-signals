"""base — abstract base class for all trading strategies.

Every strategy receives a MultiIndex ``(ticker, date)`` price DataFrame and
returns a flat DataFrame that includes the original price columns plus any
derived signal columns.

Typical usage::

    class MyStrategy(BaseStrategy):
        def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base for all equity signal strategies.

    Subclasses must implement :meth:`compute`.  The :attr:`name` property is
    derived automatically from the concrete class name (lowercased) and can be
    used as a ``strategy`` column value in output DataFrames.
    """

    @property
    def name(self) -> str:
        """Strategy identifier derived from the concrete class name (lowercased)."""
        return type(self).__name__.lower()

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute signals for all tickers in *prices*.

        Args:
            prices: MultiIndex DataFrame with levels ``(ticker, date)`` and at
                least a ``close`` column.

        Returns:
            Flat DataFrame (one row per ticker-date) with signal columns
            appended.  Must never raise for individual ticker failures — bad
            tickers should return NaN rows.
        """
