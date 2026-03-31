"""Tests for the yfinance fallback logic in AlpacaLoader.get_ohlcv.

Strategy
--------
All Alpaca SDK calls are patched at the ``_call`` method level so the real
network is never touched.  The ``fetch_ohlcv`` yfinance helper is patched at
the point it is imported inside ``alpaca_loader`` so we can control exactly
what it returns without touching yfinance.

Test layout
-----------
* ``TestAlpacaSuccess``        — Alpaca returns full data; yfinance never called.
* ``TestFullFallback``         — Alpaca raises; all tickers routed to yfinance.
* ``TestPartialMissingFallback``— Alpaca returns 3/5; yfinance fetches the 2 missing.
* ``TestBothFail``             — both sources fail; empty DataFrame + WARNING.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from equity_signals.data.alpaca_loader import AlpacaLoader
from equity_signals.exceptions import AlpacaAPIError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _make_ohlcv(tickers: list[str], n_days: int = 5) -> pd.DataFrame:
    """Return a fake MultiIndex OHLCV DataFrame for *tickers*."""
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=n_days).date.tolist()
    for ticker in tickers:
        for d in dates:
            rows.append({
                "ticker": ticker,
                "date": d,
                "open": 100.0, "high": 105.0, "low": 95.0,
                "close": 102.0, "volume": 1_000_000.0,
            })
    df = pd.DataFrame(rows)
    df = df.set_index(["ticker", "date"])
    return df[_OHLCV_COLS]


def _alpaca_loader_no_init() -> AlpacaLoader:
    """Return an AlpacaLoader whose SDK client is a MagicMock (no credentials needed)."""
    loader = AlpacaLoader.__new__(AlpacaLoader)
    loader._api_key = "fake"
    loader._secret_key = "fake"
    loader._base_url = "https://paper-api.alpaca.markets"
    loader._ffill_limit = 5
    loader.is_paper = True
    loader._client = MagicMock()
    return loader


# ---------------------------------------------------------------------------
# TestAlpacaSuccess
# ---------------------------------------------------------------------------


class TestAlpacaSuccess:
    """Alpaca returns full data → yfinance is never invoked."""

    def test_returns_alpaca_data(self) -> None:
        loader = _alpaca_loader_no_init()
        expected = _make_ohlcv(["AAPL", "MSFT"])

        with patch.object(loader, "_fetch_alpaca", return_value=expected) as mock_alpaca, \
             patch("equity_signals.data.alpaca_loader.fetch_ohlcv") as mock_yf:
            result = loader.get_ohlcv(["AAPL", "MSFT"], days=60)

        mock_alpaca.assert_called_once()
        mock_yf.assert_not_called()
        pd.testing.assert_frame_equal(result, expected)

    def test_no_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        loader = _alpaca_loader_no_init()
        expected = _make_ohlcv(["AAPL"])

        with patch.object(loader, "_fetch_alpaca", return_value=expected), \
             patch("equity_signals.data.alpaca_loader.fetch_ohlcv"), \
             caplog.at_level(logging.WARNING, logger="equity_signals.data.alpaca_loader"):
            loader.get_ohlcv(["AAPL"], days=60)

        assert not any("fallback" in m.lower() or "missing" in m.lower()
                       for m in caplog.messages)


# ---------------------------------------------------------------------------
# TestFullFallback
# ---------------------------------------------------------------------------


class TestFullFallback:
    """Alpaca raises an exception → all tickers fetched via yfinance."""

    def test_yfinance_called_for_all_tickers(self) -> None:
        loader = _alpaca_loader_no_init()
        yf_data = _make_ohlcv(["AAPL", "MSFT"])

        with patch.object(loader, "_fetch_alpaca", side_effect=AlpacaAPIError(["AAPL", "MSFT"], 500, "err")), \
             patch("equity_signals.data.alpaca_loader.fetch_ohlcv", return_value=yf_data) as mock_yf:
            result = loader.get_ohlcv(["AAPL", "MSFT"], days=60)

        mock_yf.assert_called_once_with(["AAPL", "MSFT"], 60)
        pd.testing.assert_frame_equal(result, yf_data)

    def test_warning_logged_on_alpaca_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging
        loader = _alpaca_loader_no_init()

        with patch.object(loader, "_fetch_alpaca", side_effect=RuntimeError("timeout")), \
             patch("equity_signals.data.alpaca_loader.fetch_ohlcv", return_value=_make_ohlcv(["AAPL"])), \
             caplog.at_level(logging.WARNING, logger="equity_signals.data.alpaca_loader"):
            loader.get_ohlcv(["AAPL"], days=60)

        assert any("falling back to yfinance" in m for m in caplog.messages)

    def test_returns_dataframe_not_raises(self) -> None:
        loader = _alpaca_loader_no_init()

        with patch.object(loader, "_fetch_alpaca", side_effect=RuntimeError("boom")), \
             patch("equity_signals.data.alpaca_loader.fetch_ohlcv", return_value=_make_ohlcv(["ZZZ"])):
            result = loader.get_ohlcv(["ZZZ"], days=60)

        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# TestPartialMissingFallback
# ---------------------------------------------------------------------------


class TestPartialMissingFallback:
    """Alpaca returns 3/5 tickers → yfinance called only for the 2 missing."""

    def _setup(self):
        all_tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        alpaca_tickers = ["AAA", "BBB", "CCC"]
        missing_tickers = ["DDD", "EEE"]
        return all_tickers, alpaca_tickers, missing_tickers

    def test_yfinance_called_only_for_missing(self) -> None:
        loader = _alpaca_loader_no_init()
        all_tickers, alpaca_tickers, missing_tickers = self._setup()

        alpaca_data = _make_ohlcv(alpaca_tickers)
        yf_data = _make_ohlcv(missing_tickers)

        with patch.object(loader, "_fetch_alpaca", return_value=alpaca_data), \
             patch("equity_signals.data.alpaca_loader.fetch_ohlcv", return_value=yf_data) as mock_yf:
            loader.get_ohlcv(all_tickers, days=60)

        called_tickers = set(mock_yf.call_args[0][0])
        assert called_tickers == set(missing_tickers)

    def test_result_contains_all_tickers(self) -> None:
        loader = _alpaca_loader_no_init()
        all_tickers, alpaca_tickers, missing_tickers = self._setup()

        alpaca_data = _make_ohlcv(alpaca_tickers)
        yf_data = _make_ohlcv(missing_tickers)

        with patch.object(loader, "_fetch_alpaca", return_value=alpaca_data), \
             patch("equity_signals.data.alpaca_loader.fetch_ohlcv", return_value=yf_data):
            result = loader.get_ohlcv(all_tickers, days=60)

        returned_tickers = set(result.index.get_level_values("ticker").unique())
        assert returned_tickers == set(all_tickers)

    def test_warning_logged_for_missing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging
        loader = _alpaca_loader_no_init()
        all_tickers, alpaca_tickers, missing_tickers = self._setup()

        with patch.object(loader, "_fetch_alpaca", return_value=_make_ohlcv(alpaca_tickers)), \
             patch("equity_signals.data.alpaca_loader.fetch_ohlcv", return_value=_make_ohlcv(missing_tickers)), \
             caplog.at_level(logging.WARNING, logger="equity_signals.data.alpaca_loader"):
            loader.get_ohlcv(all_tickers, days=60)

        assert any("missing" in m.lower() for m in caplog.messages)


# ---------------------------------------------------------------------------
# TestBothFail
# ---------------------------------------------------------------------------


class TestBothFail:
    """Both Alpaca and yfinance fail → empty DataFrame returned, WARNING logged."""

    def test_returns_empty_dataframe(self) -> None:
        loader = _alpaca_loader_no_init()
        empty = pd.DataFrame(
            columns=_OHLCV_COLS,
            index=pd.MultiIndex.from_tuples([], names=["ticker", "date"]),
        )

        with patch.object(loader, "_fetch_alpaca", side_effect=RuntimeError("alpaca down")), \
             patch("equity_signals.data.alpaca_loader.fetch_ohlcv", return_value=empty):
            result = loader.get_ohlcv(["FAIL"], days=60)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_does_not_raise(self) -> None:
        loader = _alpaca_loader_no_init()
        empty = pd.DataFrame(
            columns=_OHLCV_COLS,
            index=pd.MultiIndex.from_tuples([], names=["ticker", "date"]),
        )

        with patch.object(loader, "_fetch_alpaca", side_effect=RuntimeError("alpaca down")), \
             patch("equity_signals.data.alpaca_loader.fetch_ohlcv", return_value=empty):
            # Must not raise
            loader.get_ohlcv(["FAIL"], days=60)

    def test_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        loader = _alpaca_loader_no_init()
        empty = pd.DataFrame(
            columns=_OHLCV_COLS,
            index=pd.MultiIndex.from_tuples([], names=["ticker", "date"]),
        )

        with patch.object(loader, "_fetch_alpaca", side_effect=RuntimeError("alpaca down")), \
             patch("equity_signals.data.alpaca_loader.fetch_ohlcv", return_value=empty), \
             caplog.at_level(logging.WARNING, logger="equity_signals.data.alpaca_loader"):
            loader.get_ohlcv(["FAIL"], days=60)

        assert any("falling back to yfinance" in m for m in caplog.messages)
