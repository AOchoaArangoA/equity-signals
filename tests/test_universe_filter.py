"""Tests for equity_signals.universe.universe_filter.

Strategy
--------
:class:`UniverseFilter` accepts a ``loader=`` constructor argument, so every
test injects a :class:`unittest.mock.MagicMock` instead of making real HTTP
calls.

Test layout
-----------
* ``make_fundamentals`` — shared helper that builds a clean FMP DataFrame.
* Fixtures — a default ``FilterConfig`` and a pre-wired ``UniverseFilter``.
* Test classes group related assertions:
  - ``TestFilterConfig``       — dataclass validation
  - ``TestMidcapFilter``       — stage 1
  - ``TestSectorFilter``       — stage 2
  - ``TestRoeFilter``          — stage 3 (value-trap exclusion)
  - ``TestPbRanking``          — stage 4 (ranking, percentile, ties, NaN)
  - ``TestEdgeCases``          — empty inputs, full pipeline propagation
  - ``TestOutputContract``     — column presence and sort order guarantees
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from equity_signals.data.yfinance_loader import YFinanceLoader
from equity_signals.exceptions import FMPRateLimitError, FMPResponseError
from equity_signals.universe.universe_filter import (
    OUTPUT_COLS,
    FilterConfig,
    UniverseFilter,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def make_fundamentals(rows: list[dict]) -> pd.DataFrame:
    """Build a fake :meth:`FMPLoader.get_fundamentals` response DataFrame.

    Missing fields default to:
        - ``updated_at`` → ``_NOW``
        - ``roe``        → ``0.15``  (positive, so ROE filter passes by default)

    Args:
        rows: List of dicts with any subset of the FMPLoader output columns.

    Returns:
        DataFrame with columns ``[ticker, market_cap, pb_ratio, roe, sector, updated_at]``.
    """
    for row in rows:
        row.setdefault("updated_at", _NOW)
        row.setdefault("roe", 0.15)

    return pd.DataFrame(
        rows,
        columns=["ticker", "market_cap", "pb_ratio", "roe", "sector", "updated_at"],
    )


def make_loader(rows: list[dict]) -> MagicMock:
    """Return a MagicMock YFinanceLoader whose ``get_fundamentals`` returns *rows*."""
    loader = MagicMock(spec=YFinanceLoader)
    loader.get_fundamentals.return_value = make_fundamentals(rows)
    return loader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def midcap_config() -> FilterConfig:
    """FilterConfig with default midcap bounds, all sectors, percentile=100."""
    return FilterConfig(
        midcap_min=300_000_000,
        midcap_max=2_000_000_000,
        sectors=[],
        pb_percentile=100,
    )


@pytest.fixture()
def tech_industrials_config() -> FilterConfig:
    """FilterConfig restricted to Technology and Industrials, percentile=100."""
    return FilterConfig(
        midcap_min=300_000_000,
        midcap_max=2_000_000_000,
        sectors=["Technology", "Industrials"],
        pb_percentile=100,
    )


# ---------------------------------------------------------------------------
# TestFilterConfig
# ---------------------------------------------------------------------------


class TestFilterConfig:
    """Dataclass construction and validation."""

    def test_defaults_are_applied(self) -> None:
        cfg = FilterConfig()
        assert cfg.midcap_min == 300_000_000
        assert cfg.midcap_max == 2_000_000_000
        assert cfg.sectors == []
        assert cfg.pb_percentile == 30

    def test_percentile_100_is_valid(self) -> None:
        assert FilterConfig(pb_percentile=100).pb_percentile == 100

    def test_percentile_1_is_valid(self) -> None:
        assert FilterConfig(pb_percentile=1).pb_percentile == 1

    def test_percentile_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="pb_percentile"):
            FilterConfig(pb_percentile=0)

    def test_percentile_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="pb_percentile"):
            FilterConfig(pb_percentile=-10)

    def test_percentile_above_100_raises(self) -> None:
        with pytest.raises(ValueError, match="pb_percentile"):
            FilterConfig(pb_percentile=101)


# ---------------------------------------------------------------------------
# TestMidcapFilter
# ---------------------------------------------------------------------------


class TestMidcapFilter:
    """Stage 1: market-cap filter."""

    def test_below_min_excluded(self, midcap_config: FilterConfig) -> None:
        loader = make_loader([
            {"ticker": "SMALL", "market_cap": 100_000_000, "pb_ratio": 1.0, "sector": "Technology"},
        ])
        assert UniverseFilter(midcap_config, loader=loader).run(["SMALL"]).empty

    def test_above_max_excluded(self, midcap_config: FilterConfig) -> None:
        loader = make_loader([
            {"ticker": "GIANT", "market_cap": 5_000_000_000, "pb_ratio": 1.0, "sector": "Technology"},
        ])
        assert UniverseFilter(midcap_config, loader=loader).run(["GIANT"]).empty

    def test_at_lower_boundary_included(self, midcap_config: FilterConfig) -> None:
        loader = make_loader([
            {"ticker": "EDGE", "market_cap": 300_000_000, "pb_ratio": 1.0, "sector": "Technology"},
        ])
        assert "EDGE" in UniverseFilter(midcap_config, loader=loader).run(["EDGE"])["ticker"].values

    def test_at_upper_boundary_included(self, midcap_config: FilterConfig) -> None:
        loader = make_loader([
            {"ticker": "EDGE", "market_cap": 2_000_000_000, "pb_ratio": 1.0, "sector": "Technology"},
        ])
        assert "EDGE" in UniverseFilter(midcap_config, loader=loader).run(["EDGE"])["ticker"].values

    def test_within_range_included(self, midcap_config: FilterConfig) -> None:
        loader = make_loader([
            {"ticker": "MID", "market_cap": 800_000_000, "pb_ratio": 1.5, "sector": "Energy"},
        ])
        assert "MID" in UniverseFilter(midcap_config, loader=loader).run(["MID"])["ticker"].values

    def test_missing_market_cap_excluded(self, midcap_config: FilterConfig) -> None:
        loader = make_loader([
            {"ticker": "NOMCAP", "market_cap": None, "pb_ratio": 1.0, "sector": "Technology"},
        ])
        assert UniverseFilter(midcap_config, loader=loader).run(["NOMCAP"]).empty

    def test_mixed_caps_only_midcap_survives(self, midcap_config: FilterConfig) -> None:
        loader = make_loader([
            {"ticker": "SMALL", "market_cap": 50_000_000,    "pb_ratio": 1.0, "sector": "Technology"},
            {"ticker": "MID",   "market_cap": 600_000_000,   "pb_ratio": 2.0, "sector": "Technology"},
            {"ticker": "LARGE", "market_cap": 10_000_000_000,"pb_ratio": 3.0, "sector": "Technology"},
        ])
        df = UniverseFilter(midcap_config, loader=loader).run(["SMALL", "MID", "LARGE"])
        assert list(df["ticker"]) == ["MID"]


# ---------------------------------------------------------------------------
# TestSectorFilter
# ---------------------------------------------------------------------------


class TestSectorFilter:
    """Stage 2: sector filter."""

    def test_wrong_sector_excluded(self, tech_industrials_config: FilterConfig) -> None:
        loader = make_loader([
            {"ticker": "OIL", "market_cap": 500_000_000, "pb_ratio": 1.0, "sector": "Energy"},
        ])
        assert UniverseFilter(tech_industrials_config, loader=loader).run(["OIL"]).empty

    def test_correct_sector_included(self, tech_industrials_config: FilterConfig) -> None:
        loader = make_loader([
            {"ticker": "TECH", "market_cap": 500_000_000, "pb_ratio": 1.0, "sector": "Technology"},
        ])
        assert "TECH" in UniverseFilter(tech_industrials_config, loader=loader).run(["TECH"])["ticker"].values

    def test_none_sector_excluded_when_filter_active(
        self, tech_industrials_config: FilterConfig
    ) -> None:
        loader = make_loader([
            {"ticker": "UNKNOWN", "market_cap": 500_000_000, "pb_ratio": 1.0, "sector": None},
        ])
        assert UniverseFilter(tech_industrials_config, loader=loader).run(["UNKNOWN"]).empty

    def test_empty_sectors_accepts_all(self, midcap_config: FilterConfig) -> None:
        loader = make_loader([
            {"ticker": "A", "market_cap": 500_000_000, "pb_ratio": 1.0, "sector": "Energy"},
            {"ticker": "B", "market_cap": 600_000_000, "pb_ratio": 2.0, "sector": "Healthcare"},
            {"ticker": "C", "market_cap": 700_000_000, "pb_ratio": 3.0, "sector": "Technology"},
        ])
        df = UniverseFilter(midcap_config, loader=loader).run(["A", "B", "C"])
        assert set(df["ticker"]) == {"A", "B", "C"}

    def test_multiple_allowed_sectors_both_kept(
        self, tech_industrials_config: FilterConfig
    ) -> None:
        loader = make_loader([
            {"ticker": "TECH", "market_cap": 500_000_000, "pb_ratio": 1.0, "sector": "Technology"},
            {"ticker": "IND",  "market_cap": 600_000_000, "pb_ratio": 2.0, "sector": "Industrials"},
            {"ticker": "OIL",  "market_cap": 700_000_000, "pb_ratio": 3.0, "sector": "Energy"},
        ])
        df = UniverseFilter(tech_industrials_config, loader=loader).run(["TECH", "IND", "OIL"])
        assert set(df["ticker"]) == {"TECH", "IND"}


# ---------------------------------------------------------------------------
# TestRoeFilter
# ---------------------------------------------------------------------------


class TestRoeFilter:
    """Stage 3: ROE > 0 value-trap exclusion filter."""

    def _filter(self, rows: list[dict]) -> pd.DataFrame:
        """Helper: pass-through midcap and sector, run full pipeline."""
        cfg = FilterConfig(
            midcap_min=0,
            midcap_max=10_000_000_000_000,
            sectors=[],
            pb_percentile=100,
        )
        return UniverseFilter(cfg, loader=make_loader(rows)).run(
            [r["ticker"] for r in rows]
        )

    def test_negative_roe_excluded(self) -> None:
        df = self._filter([
            {"ticker": "TRAP", "market_cap": 500e6, "pb_ratio": 0.5, "sector": "Energy", "roe": -0.10},
        ])
        assert df.empty

    def test_zero_roe_excluded(self) -> None:
        df = self._filter([
            {"ticker": "ZERO", "market_cap": 500e6, "pb_ratio": 0.5, "sector": "Energy", "roe": 0.0},
        ])
        assert df.empty

    def test_positive_roe_included(self) -> None:
        df = self._filter([
            {"ticker": "GOOD", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Energy", "roe": 0.01},
        ])
        assert "GOOD" in df["ticker"].values

    def test_nan_roe_excluded(self) -> None:
        df = self._filter([
            {"ticker": "NOROE", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Technology", "roe": None},
        ])
        assert df.empty

    def test_mixed_roe_only_positive_survives(self) -> None:
        df = self._filter([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Energy", "roe": -0.05},
            {"ticker": "B", "market_cap": 600e6, "pb_ratio": 2.0, "sector": "Energy", "roe": 0.0},
            {"ticker": "C", "market_cap": 700e6, "pb_ratio": 3.0, "sector": "Energy", "roe": 0.20},
        ])
        assert list(df["ticker"]) == ["C"]

    def test_roe_filter_does_not_affect_pb_ranking(self) -> None:
        """After ROE filter, P/B ranking of surviving tickers must be correct."""
        df = self._filter([
            {"ticker": "CHEAP", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Tech", "roe": 0.15},
            {"ticker": "PRICEY","market_cap": 600e6, "pb_ratio": 3.0, "sector": "Tech", "roe": 0.10},
            {"ticker": "TRAP",  "market_cap": 700e6, "pb_ratio": 0.3, "sector": "Tech", "roe": -0.20},
        ])
        rank_map = df.set_index("ticker")["pb_rank_sector"].to_dict()
        assert rank_map["CHEAP"] == 1
        assert rank_map["PRICEY"] == 2
        assert "TRAP" not in rank_map


# ---------------------------------------------------------------------------
# TestPbRanking
# ---------------------------------------------------------------------------


class TestPbRanking:
    """Stage 4: intra-sector P/B ranking, percentile cut, and value_signal."""

    def _filter(
        self,
        rows: list[dict],
        *,
        pb_percentile: int = 100,
        sectors: list[str] | None = None,
    ) -> pd.DataFrame:
        cfg = FilterConfig(
            midcap_min=0,
            midcap_max=10_000_000_000_000,
            sectors=sectors or [],
            pb_percentile=pb_percentile,
        )
        return UniverseFilter(cfg, loader=make_loader(rows)).run(
            [r["ticker"] for r in rows]
        )

    def test_lowest_pb_gets_rank_1(self) -> None:
        df = self._filter([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 3.0, "sector": "Technology"},
            {"ticker": "B", "market_cap": 600e6, "pb_ratio": 1.0, "sector": "Technology"},
            {"ticker": "C", "market_cap": 700e6, "pb_ratio": 2.0, "sector": "Technology"},
        ])
        assert df[df["ticker"] == "B"].iloc[0]["pb_rank_sector"] == 1

    def test_ranks_assigned_ascending(self) -> None:
        df = self._filter([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Technology"},
            {"ticker": "B", "market_cap": 600e6, "pb_ratio": 2.0, "sector": "Technology"},
            {"ticker": "C", "market_cap": 700e6, "pb_ratio": 3.0, "sector": "Technology"},
        ])
        rank_map = df.set_index("ticker")["pb_rank_sector"].to_dict()
        assert rank_map == {"A": 1, "B": 2, "C": 3}

    def test_pb_ranking_independent_per_sector(self) -> None:
        df = self._filter([
            {"ticker": "T1", "market_cap": 500e6, "pb_ratio": 5.0, "sector": "Technology"},
            {"ticker": "T2", "market_cap": 600e6, "pb_ratio": 6.0, "sector": "Technology"},
            {"ticker": "I1", "market_cap": 700e6, "pb_ratio": 1.0, "sector": "Industrials"},
            {"ticker": "I2", "market_cap": 800e6, "pb_ratio": 2.0, "sector": "Industrials"},
        ])
        rank_map = df.set_index("ticker")["pb_rank_sector"].to_dict()
        assert rank_map["T1"] == 1
        assert rank_map["T2"] == 2
        assert rank_map["I1"] == 1
        assert rank_map["I2"] == 2

    def test_tied_pb_share_lowest_rank(self) -> None:
        df = self._filter([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Technology"},
            {"ticker": "B", "market_cap": 600e6, "pb_ratio": 1.0, "sector": "Technology"},
            {"ticker": "C", "market_cap": 700e6, "pb_ratio": 3.0, "sector": "Technology"},
        ])
        rank_map = df.set_index("ticker")["pb_rank_sector"].to_dict()
        assert rank_map["A"] == 1
        assert rank_map["B"] == 1
        assert rank_map["C"] == 3

    def test_nan_pb_ranked_last(self) -> None:
        df = self._filter([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 1.0,  "sector": "Technology"},
            {"ticker": "B", "market_cap": 600e6, "pb_ratio": None, "sector": "Technology"},
            {"ticker": "C", "market_cap": 700e6, "pb_ratio": 2.0,  "sector": "Technology"},
        ])
        rank_map = df.set_index("ticker")["pb_rank_sector"].to_dict()
        assert rank_map["A"] < rank_map["B"]
        assert rank_map["C"] < rank_map["B"]

    def test_percentile_50_keeps_cheaper_half(self) -> None:
        df = self._filter([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Technology"},
            {"ticker": "B", "market_cap": 600e6, "pb_ratio": 2.0, "sector": "Technology"},
            {"ticker": "C", "market_cap": 700e6, "pb_ratio": 3.0, "sector": "Technology"},
            {"ticker": "D", "market_cap": 800e6, "pb_ratio": 4.0, "sector": "Technology"},
        ], pb_percentile=50)
        signal = df.set_index("ticker")["value_signal"].to_dict()
        assert signal["A"] == True
        assert signal["B"] == True
        assert signal["C"] == False
        assert signal["D"] == False

    def test_percentile_100_all_true(self) -> None:
        df = self._filter([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Energy"},
            {"ticker": "B", "market_cap": 600e6, "pb_ratio": 9.0, "sector": "Energy"},
        ], pb_percentile=100)
        assert df["value_signal"].all()

    def test_percentile_guarantees_at_least_one_per_sector(self) -> None:
        df = self._filter([
            {"ticker": "SOLO", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Energy"},
        ], pb_percentile=1)
        assert df["value_signal"].iloc[0] == True

    def test_percentile_applied_independently_per_sector(self) -> None:
        rows = [
            {"ticker": "T1", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Technology"},
            {"ticker": "T2", "market_cap": 600e6, "pb_ratio": 2.0, "sector": "Technology"},
            {"ticker": "T3", "market_cap": 700e6, "pb_ratio": 3.0, "sector": "Technology"},
            {"ticker": "T4", "market_cap": 800e6, "pb_ratio": 4.0, "sector": "Technology"},
            {"ticker": "I1", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Industrials"},
            {"ticker": "I2", "market_cap": 600e6, "pb_ratio": 2.0, "sector": "Industrials"},
        ]
        df = self._filter(rows, pb_percentile=25)
        signal = df.set_index("ticker")["value_signal"].to_dict()
        assert signal["T1"] == True
        assert signal["I1"] == True
        for t in ("T2", "T3", "T4", "I2"):
            assert signal[t] == False

    def test_value_signal_false_for_expensive_tickers(self) -> None:
        df = self._filter([
            {"ticker": "CHEAP",    "market_cap": 500e6, "pb_ratio": 0.5, "sector": "Energy"},
            {"ticker": "EXPENSIVE","market_cap": 600e6, "pb_ratio": 9.9, "sector": "Energy"},
        ], pb_percentile=50)
        signal = df.set_index("ticker")["value_signal"].to_dict()
        assert signal["CHEAP"] == True
        assert signal["EXPENSIVE"] == False


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Empty inputs, all-filtered universes, and exception propagation."""

    def test_empty_ticker_list_returns_empty_df(self) -> None:
        loader = MagicMock(spec=YFinanceLoader)
        df = UniverseFilter(loader=loader).run([])
        assert df.empty
        assert list(df.columns) == OUTPUT_COLS
        loader.get_fundamentals.assert_not_called()

    def test_fmp_returns_empty_df(self) -> None:
        loader = MagicMock(spec=YFinanceLoader)
        loader.get_fundamentals.return_value = pd.DataFrame(
            columns=["ticker", "market_cap", "pb_ratio", "roe", "sector", "updated_at"]
        )
        df = UniverseFilter(loader=loader).run(["A", "B"])
        assert df.empty
        assert list(df.columns) == OUTPUT_COLS

    def test_all_filtered_by_midcap_returns_empty(self) -> None:
        loader = make_loader([
            {"ticker": "NANO", "market_cap": 1_000, "pb_ratio": 1.0, "sector": "Technology"},
        ])
        cfg = FilterConfig(midcap_min=300_000_000, midcap_max=2_000_000_000, pb_percentile=100)
        df = UniverseFilter(cfg, loader=loader).run(["NANO"])
        assert df.empty
        assert list(df.columns) == OUTPUT_COLS

    def test_all_filtered_by_sector_returns_empty(self) -> None:
        loader = make_loader([
            {"ticker": "OIL", "market_cap": 500_000_000, "pb_ratio": 1.0, "sector": "Energy"},
        ])
        cfg = FilterConfig(
            midcap_min=0, midcap_max=10_000_000_000_000,
            sectors=["Technology"],
            pb_percentile=100,
        )
        df = UniverseFilter(cfg, loader=loader).run(["OIL"])
        assert df.empty
        assert list(df.columns) == OUTPUT_COLS

    def test_all_filtered_by_roe_returns_empty(self) -> None:
        loader = make_loader([
            {"ticker": "TRAP", "market_cap": 500_000_000, "pb_ratio": 0.5,
             "sector": "Energy", "roe": -0.30},
        ])
        cfg = FilterConfig(
            midcap_min=0, midcap_max=10_000_000_000_000, pb_percentile=100
        )
        df = UniverseFilter(cfg, loader=loader).run(["TRAP"])
        assert df.empty
        assert list(df.columns) == OUTPUT_COLS

    def test_fmp_rate_limit_propagates(self) -> None:
        loader = MagicMock(spec=YFinanceLoader)
        loader.get_fundamentals.side_effect = FMPRateLimitError("AAA", 3)
        with pytest.raises(FMPRateLimitError):
            UniverseFilter(loader=loader).run(["AAA"])

    def test_fmp_response_error_propagates(self) -> None:
        loader = MagicMock(spec=YFinanceLoader)
        loader.get_fundamentals.side_effect = FMPResponseError("BBB", 500)
        with pytest.raises(FMPResponseError):
            UniverseFilter(loader=loader).run(["BBB"])

    def test_single_ticker_survives_full_pipeline(self) -> None:
        loader = make_loader([
            {"ticker": "SOLO", "market_cap": 500_000_000, "pb_ratio": 1.5,
             "sector": "Energy", "roe": 0.12},
        ])
        cfg = FilterConfig(
            midcap_min=300_000_000, midcap_max=2_000_000_000,
            sectors=["Energy"],
            pb_percentile=100,
        )
        df = UniverseFilter(cfg, loader=loader).run(["SOLO"])
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "SOLO"
        assert df.iloc[0]["value_signal"] == True

    def test_loader_called_with_exact_tickers(self, mocker) -> None:
        loader = MagicMock(spec=YFinanceLoader)
        loader.get_fundamentals.return_value = pd.DataFrame(
            columns=["ticker", "market_cap", "pb_ratio", "roe", "sector", "updated_at"]
        )
        tickers = ["AAPL", "MSFT", "GOOG"]
        UniverseFilter(loader=loader).run(tickers)
        loader.get_fundamentals.assert_called_once_with(tickers)


# ---------------------------------------------------------------------------
# TestOutputContract
# ---------------------------------------------------------------------------


class TestOutputContract:
    """Column presence, types, and sort-order guarantees on the returned DataFrame."""

    def _run_with_rows(self, rows: list[dict], **cfg_kwargs) -> pd.DataFrame:
        cfg = FilterConfig(
            midcap_min=0,
            midcap_max=10_000_000_000_000,
            pb_percentile=100,
            **cfg_kwargs,
        )
        return UniverseFilter(cfg, loader=make_loader(rows)).run(
            [r["ticker"] for r in rows]
        )

    def test_output_columns_exact(self) -> None:
        df = self._run_with_rows([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Energy"},
        ])
        assert list(df.columns) == OUTPUT_COLS

    def test_empty_result_has_correct_columns(self) -> None:
        loader = MagicMock(spec=YFinanceLoader)
        loader.get_fundamentals.return_value = pd.DataFrame(
            columns=["ticker", "market_cap", "pb_ratio", "roe", "sector", "updated_at"]
        )
        df = UniverseFilter(loader=loader).run(["X"])
        assert list(df.columns) == OUTPUT_COLS

    def test_sorted_by_sector_then_pb_rank(self) -> None:
        rows = [
            {"ticker": "T2", "market_cap": 600e6, "pb_ratio": 2.0, "sector": "Technology"},
            {"ticker": "I1", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Industrials"},
            {"ticker": "T1", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Technology"},
            {"ticker": "I2", "market_cap": 600e6, "pb_ratio": 3.0, "sector": "Industrials"},
        ]
        df = self._run_with_rows(rows)
        assert list(df["sector"]) == ["Industrials", "Industrials", "Technology", "Technology"]
        for sector in ("Industrials", "Technology"):
            ranks = df[df["sector"] == sector]["pb_rank_sector"].tolist()
            assert ranks == sorted(ranks)

    def test_value_signal_is_bool(self) -> None:
        df = self._run_with_rows([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Energy"},
        ])
        assert df["value_signal"].dtype == bool or df["value_signal"].map(type).eq(bool).all()

    def test_pb_rank_sector_is_integer(self) -> None:
        df = self._run_with_rows([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Energy"},
        ])
        assert pd.api.types.is_integer_dtype(df["pb_rank_sector"])

    def test_updated_at_not_in_output(self) -> None:
        df = self._run_with_rows([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Energy"},
        ])
        assert "updated_at" not in df.columns

    def test_roe_column_present_in_output(self) -> None:
        df = self._run_with_rows([
            {"ticker": "A", "market_cap": 500e6, "pb_ratio": 1.0, "sector": "Energy", "roe": 0.15},
        ])
        assert "roe" in df.columns
