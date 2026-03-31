"""Tests for equity_signals.data.yfinance_loader.

Strategy
--------
``yfinance.Ticker(t).info`` is patched at the class level so no network calls
are made.  Every test constructs a :class:`YFinanceLoader` with
``cache_dir`` pointing at a ``tmp_path`` so cache files are isolated and
cleaned up automatically by pytest.

Test layout
-----------
* ``TestGetFundamentals``   — happy path, NaN handling, full-ticker failure
* ``TestCacheBehaviour``    — cache hit skips API; stale cache triggers re-fetch
* ``TestOutputSchema``      — column names, dtypes, row count guarantees
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from equity_signals.data.yfinance_loader import (
    YFinanceLoader,
    _FUNDAMENTALS_COLS,
    _CACHE_PREFIX,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_INFO: dict = {
    "marketCap": 500_000_000,
    "priceToBook": 1.5,
    "returnOnEquity": 0.12,
    "sector": "Technology",
}

_MISSING_PB_INFO: dict = {
    "marketCap": 800_000_000,
    "priceToBook": None,
    "returnOnEquity": 0.08,
    "sector": "Industrials",
}

_EMPTY_INFO: dict = {}


def _make_loader(tmp_path: Path, **kwargs) -> YFinanceLoader:
    """Return a YFinanceLoader that writes cache to *tmp_path*."""
    kwargs.setdefault("cache_ttl_days", 7)
    return YFinanceLoader(cache_dir=tmp_path, **kwargs)


def _patch_ticker(info_map: dict[str, dict]):
    """Return a context-manager that patches yfinance.Ticker.info per symbol.

    Args:
        info_map: Mapping of ticker symbol → ``info`` dict to return.
            A ticker mapped to an exception *instance* will raise on access.
    """
    original_init = None

    class FakeTicker:
        def __init__(self, symbol: str, **kwargs) -> None:
            self._symbol = symbol

        @property
        def info(self) -> dict:
            result = info_map.get(self._symbol, {})
            if isinstance(result, Exception):
                raise result
            return result

    return patch("equity_signals.data.yfinance_loader.yf.Ticker", FakeTicker)


# ---------------------------------------------------------------------------
# TestGetFundamentals
# ---------------------------------------------------------------------------


class TestGetFundamentals:
    """Happy path and error-handling behaviour of get_fundamentals()."""

    def test_returns_correct_columns(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        with _patch_ticker({"AAPL": _GOOD_INFO}):
            df = loader.get_fundamentals(["AAPL"])
        assert list(df.columns) == _FUNDAMENTALS_COLS

    def test_extracts_all_fields_correctly(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        with _patch_ticker({"AAPL": _GOOD_INFO}):
            df = loader.get_fundamentals(["AAPL"])
        row = df[df["ticker"] == "AAPL"].iloc[0]
        assert row["market_cap"] == 500_000_000
        assert row["pb_ratio"] == 1.5
        assert row["roe"] == 0.12
        assert row["sector"] == "Technology"

    def test_one_row_per_ticker(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        tickers = ["AAPL", "MSFT", "GOOG"]
        info_map = {t: _GOOD_INFO for t in tickers}
        with _patch_ticker(info_map):
            df = loader.get_fundamentals(tickers)
        assert len(df) == len(tickers)
        assert set(df["ticker"]) == set(tickers)

    def test_missing_pb_ratio_is_nan(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        with _patch_ticker({"GE": _MISSING_PB_INFO}):
            df = loader.get_fundamentals(["GE"])
        assert pd.isna(df.iloc[0]["pb_ratio"])

    def test_empty_info_produces_nan_row(self, tmp_path: Path) -> None:
        """A ticker returning {} should give a row with all-NaN numerics."""
        loader = _make_loader(tmp_path)
        with _patch_ticker({"UNKNOWN": _EMPTY_INFO}):
            df = loader.get_fundamentals(["UNKNOWN"])
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "UNKNOWN"
        assert pd.isna(df.iloc[0]["market_cap"])
        assert pd.isna(df.iloc[0]["pb_ratio"])
        assert pd.isna(df.iloc[0]["roe"])
        assert df.iloc[0]["sector"] is None

    def test_ticker_exception_produces_nan_row_not_crash(self, tmp_path: Path) -> None:
        """A ticker that raises must return a NaN row — never kill the batch."""
        loader = _make_loader(tmp_path)
        boom = RuntimeError("network timeout")
        with _patch_ticker({"CRASH": boom, "OK": _GOOD_INFO}):
            df = loader.get_fundamentals(["CRASH", "OK"])
        assert len(df) == 2
        crash_row = df[df["ticker"] == "CRASH"].iloc[0]
        ok_row = df[df["ticker"] == "OK"].iloc[0]
        assert pd.isna(crash_row["market_cap"])
        assert ok_row["market_cap"] == 500_000_000

    def test_mixed_good_and_missing_fields(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        with _patch_ticker({"GOOD": _GOOD_INFO, "BAD": _MISSING_PB_INFO}):
            df = loader.get_fundamentals(["GOOD", "BAD"])
        assert len(df) == 2
        assert not pd.isna(df[df["ticker"] == "GOOD"].iloc[0]["pb_ratio"])
        assert pd.isna(df[df["ticker"] == "BAD"].iloc[0]["pb_ratio"])

    def test_yfinance_na_string_treated_as_none(self, tmp_path: Path) -> None:
        """Some yfinance versions return the string 'N/A' instead of None."""
        loader = _make_loader(tmp_path)
        info = {**_GOOD_INFO, "sector": "N/A", "priceToBook": "N/A"}
        with _patch_ticker({"XYZ": info}):
            df = loader.get_fundamentals(["XYZ"])
        row = df.iloc[0]
        assert row["sector"] is None
        assert pd.isna(row["pb_ratio"])

    def test_updated_at_is_utc_datetime(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        with _patch_ticker({"AAPL": _GOOD_INFO}):
            df = loader.get_fundamentals(["AAPL"])
        ts = df.iloc[0]["updated_at"]
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None


# ---------------------------------------------------------------------------
# TestCacheBehaviour
# ---------------------------------------------------------------------------


class TestCacheBehaviour:
    """Parquet cache hit/miss logic and TTL enforcement."""

    def test_cache_hit_skips_api_calls(self, tmp_path: Path, mocker) -> None:
        """When a fresh cache file exists, yfinance must not be called."""
        loader = _make_loader(tmp_path)

        # Pre-populate cache with a fresh file dated today.
        cached_df = pd.DataFrame([{
            "ticker": "AAPL",
            "market_cap": 500_000_000.0,
            "pb_ratio": 1.5,
            "roe": 0.12,
            "sector": "Technology",
            "updated_at": datetime.now(tz=timezone.utc),
        }])
        cache_path = tmp_path / f"{_CACHE_PREFIX}{date.today().strftime('%Y%m%d')}.parquet"
        cached_df.to_parquet(cache_path, index=False)

        spy = mocker.patch("equity_signals.data.yfinance_loader.yf.Ticker")
        result = loader.get_fundamentals(["AAPL"])

        spy.assert_not_called()
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"

    def test_stale_cache_triggers_api_fetch(self, tmp_path: Path) -> None:
        """A cache file older than TTL must be ignored."""
        loader = _make_loader(tmp_path, cache_ttl_days=7)

        # Write a cache file dated 8 days ago (beyond TTL).
        stale_date = date.today() - timedelta(days=8)
        stale_path = tmp_path / f"{_CACHE_PREFIX}{stale_date.strftime('%Y%m%d')}.parquet"
        stale_df = pd.DataFrame([{
            "ticker": "STALE",
            "market_cap": 1.0,
            "pb_ratio": 1.0,
            "roe": 0.01,
            "sector": "Old",
            "updated_at": datetime.now(tz=timezone.utc),
        }])
        stale_df.to_parquet(stale_path, index=False)

        with _patch_ticker({"AAPL": _GOOD_INFO}):
            result = loader.get_fundamentals(["AAPL"])

        # Must return fresh data, not the stale cache row.
        assert "STALE" not in result["ticker"].values
        assert "AAPL" in result["ticker"].values

    def test_fresh_cache_is_written_after_fetch(self, tmp_path: Path) -> None:
        """After a successful fetch a parquet file must appear in cache_dir."""
        loader = _make_loader(tmp_path)
        with _patch_ticker({"AAPL": _GOOD_INFO}):
            loader.get_fundamentals(["AAPL"])

        today_str = date.today().strftime("%Y%m%d")
        expected = tmp_path / f"{_CACHE_PREFIX}{today_str}.parquet"
        assert expected.exists()

    def test_cache_content_matches_fetched_data(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        with _patch_ticker({"AAPL": _GOOD_INFO}):
            loader.get_fundamentals(["AAPL"])

        today_str = date.today().strftime("%Y%m%d")
        cached = pd.read_parquet(tmp_path / f"{_CACHE_PREFIX}{today_str}.parquet")
        assert cached.iloc[0]["ticker"] == "AAPL"
        assert cached.iloc[0]["pb_ratio"] == 1.5


# ---------------------------------------------------------------------------
# TestOutputSchema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """Column names, ordering, and dtype guarantees."""

    def test_columns_are_exact(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        with _patch_ticker({"A": _GOOD_INFO}):
            df = loader.get_fundamentals(["A"])
        assert list(df.columns) == _FUNDAMENTALS_COLS

    def test_empty_ticker_list_returns_empty_df_with_correct_columns(
        self, tmp_path: Path
    ) -> None:
        loader = _make_loader(tmp_path)
        with _patch_ticker({}):
            df = loader.get_fundamentals([])
        assert df.empty
        assert list(df.columns) == _FUNDAMENTALS_COLS

    def test_ticker_column_contains_input_symbols(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        tickers = ["AAPL", "MSFT"]
        with _patch_ticker({t: _GOOD_INFO for t in tickers}):
            df = loader.get_fundamentals(tickers)
        assert set(df["ticker"]) == set(tickers)

    def test_market_cap_is_numeric(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        with _patch_ticker({"AAPL": _GOOD_INFO}):
            df = loader.get_fundamentals(["AAPL"])
        assert pd.api.types.is_numeric_dtype(df["market_cap"])

    def test_pb_ratio_is_numeric(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        with _patch_ticker({"AAPL": _GOOD_INFO}):
            df = loader.get_fundamentals(["AAPL"])
        assert pd.api.types.is_numeric_dtype(df["pb_ratio"])
