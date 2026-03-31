"""Tests for equity_signals.strategies.mean_reversion.

All tests use synthetic price data — no network calls.

Test layout
-----------
* ``make_prices``     — helper that builds a MultiIndex price DataFrame.
* ``TestSignalLong``  — price drops well below MA → signal=1.
* ``TestSignalNeutral``— price stays near MA → signal=0.
* ``TestFlatPrice``   — constant price → z_score=0, signal=0.
* ``TestInsufficientData`` — fewer rows than window → NaN + warning.
* ``TestLongOnly``    — signal never equals -1.
* ``TestOutputSchema``— output columns, dtypes, and structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_signals.strategies.mean_reversion import MeanReversionStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WINDOW = 20


def make_prices(
    series: dict[str, list[float]],
    start: str = "2024-01-01",
    freq: str = "B",
) -> pd.DataFrame:
    """Build a MultiIndex ``(ticker, date)`` DataFrame with a ``close`` column.

    Args:
        series: Mapping of ticker → list of close prices.
        start: ISO date string for the first bar.
        freq: Pandas date frequency string.  ``"B"`` = business days.

    Returns:
        MultiIndex DataFrame suitable for :meth:`MeanReversionStrategy.compute`.
    """
    frames = []
    for ticker, prices in series.items():
        dates = pd.bdate_range(start=start, periods=len(prices))
        idx = pd.MultiIndex.from_arrays(
            [np.full(len(prices), ticker), dates.date],
            names=["ticker", "date"],
        )
        frames.append(pd.DataFrame({"close": prices}, index=idx))
    return pd.concat(frames).sort_index()


def _stable_then_drop(window: int, z_entry: float, drop_sigma: float = 3.0) -> list[float]:
    """Return ``window * 2`` prices: stable for ``window`` bars, then one large drop.

    The stable phase establishes a clean MA and std so the drop bar is
    guaranteed to produce ``|z| > z_entry``.
    """
    stable = [100.0] * window
    std_est = np.std(stable, ddof=1) if np.std(stable, ddof=1) > 0 else 1.0
    # After window stable bars the rolling std will be ~0; use a tiny perturbation
    # to give the rolling std something to work with, then a large drop.
    perturbed = [100.0 + (i % 3 - 1) * 0.1 for i in range(window)]  # slight noise
    drop_value = 100.0 - drop_sigma * 2.0  # well below any MA
    return perturbed + [drop_value]


# ---------------------------------------------------------------------------
# TestSignalLong
# ---------------------------------------------------------------------------


class TestSignalLong:
    """Price drops well below MA → expect signal=1 on final bar."""

    def test_last_bar_signal_is_1(self) -> None:
        """A large price drop below MA should yield signal=1."""
        # Build 30 bars: stable at 100, then drop to 70 on bar 31.
        prices_list = [100.0] * 20 + [99.0, 101.0, 100.5, 99.5, 100.0,
                                       100.2, 99.8, 100.1, 99.9, 100.0] + [70.0]
        df = make_prices({"AAA": prices_list})
        strategy = MeanReversionStrategy(window=_WINDOW, z_entry=1.5, z_exit=0.5)
        out = strategy.compute(df)

        last = out[out["ticker"] == "AAA"].iloc[-1]
        assert last["signal"] == 1, f"Expected signal=1, got {last['signal']} (z={last['z_score']:.3f})"

    def test_z_score_is_negative_on_drop(self) -> None:
        """Z-score must be negative when price is below MA."""
        prices_list = [100.0] * 20 + [70.0]
        df = make_prices({"BBB": prices_list})
        strategy = MeanReversionStrategy(window=_WINDOW, z_entry=1.5)
        out = strategy.compute(df)

        last = out[out["ticker"] == "BBB"].iloc[-1]
        assert last["z_score"] < 0, f"Expected z<0, got {last['z_score']}"


# ---------------------------------------------------------------------------
# TestSignalNeutral
# ---------------------------------------------------------------------------


class TestSignalNeutral:
    """Price stays close to MA → signal=0 throughout."""

    def test_flat_around_ma(self) -> None:
        """Prices oscillating within z_exit band should all produce signal=0."""
        # Tiny random noise: z will stay well within ±0.5
        rng = np.random.default_rng(42)
        prices_list = (100 + rng.normal(0, 0.001, 40)).tolist()
        df = make_prices({"CCC": prices_list})
        strategy = MeanReversionStrategy(window=_WINDOW, z_entry=1.5, z_exit=0.5)
        out = strategy.compute(df)

        non_nan = out[out["ticker"] == "CCC"]["signal"].dropna()
        assert (non_nan == 0).all(), f"Expected all 0, got:\n{non_nan.value_counts()}"


# ---------------------------------------------------------------------------
# TestFlatPrice
# ---------------------------------------------------------------------------


class TestFlatPrice:
    """Constant price produces zero std → z_score=0, signal=0."""

    def test_z_score_is_zero(self) -> None:
        prices_list = [50.0] * 30
        df = make_prices({"DDD": prices_list})
        strategy = MeanReversionStrategy(window=_WINDOW, z_entry=1.5)
        out = strategy.compute(df)

        non_nan = out[out["ticker"] == "DDD"]["z_score"].dropna()
        assert (non_nan == 0).all(), f"Expected all z=0, got {non_nan.tolist()}"

    def test_signal_is_zero(self) -> None:
        prices_list = [50.0] * 30
        df = make_prices({"EEE": prices_list})
        strategy = MeanReversionStrategy(window=_WINDOW, z_entry=1.5)
        out = strategy.compute(df)

        non_nan = out[out["ticker"] == "EEE"]["signal"].dropna()
        assert (non_nan == 0).all()


# ---------------------------------------------------------------------------
# TestInsufficientData
# ---------------------------------------------------------------------------


class TestInsufficientData:
    """Fewer rows than window → NaN output + warning logged."""

    def test_returns_nan_row(self) -> None:
        prices_list = [100.0] * 10  # window=20, only 10 rows
        df = make_prices({"FFF": prices_list})
        strategy = MeanReversionStrategy(window=_WINDOW, z_entry=1.5)
        out = strategy.compute(df)

        assert len(out) == 1
        assert out.iloc[0]["ticker"] == "FFF"
        assert pd.isna(out.iloc[0]["signal"])

    def test_warning_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        prices_list = [100.0] * 5
        df = make_prices({"GGG": prices_list})
        strategy = MeanReversionStrategy(window=_WINDOW, z_entry=1.5)
        import logging
        with caplog.at_level(logging.WARNING, logger="equity_signals.strategies.mean_reversion"):
            strategy.compute(df)
        assert any("GGG" in msg for msg in caplog.messages)

    def test_other_tickers_unaffected(self) -> None:
        """Insufficient ticker must not disrupt a valid sibling ticker."""
        short_prices = [100.0] * 5
        long_prices = [100.0] * 25
        df = make_prices({"SHORT": short_prices, "LONG": long_prices})
        strategy = MeanReversionStrategy(window=_WINDOW, z_entry=1.5)
        out = strategy.compute(df)

        long_rows = out[out["ticker"] == "LONG"]
        assert len(long_rows) == 25
        valid = long_rows["signal"].dropna()
        assert len(valid) > 0  # at least some valid signals for LONG


# ---------------------------------------------------------------------------
# TestLongOnly
# ---------------------------------------------------------------------------


class TestLongOnly:
    """Signal must only take values {0, 1} — never -1."""

    def test_no_short_signals_random_prices(self) -> None:
        rng = np.random.default_rng(0)
        prices_list = (100 + rng.normal(0, 5, 60)).tolist()
        df = make_prices({"HHH": prices_list})
        strategy = MeanReversionStrategy(window=_WINDOW, z_entry=1.5)
        out = strategy.compute(df)

        signals = out[out["ticker"] == "HHH"]["signal"].dropna()
        assert (signals >= 0).all(), f"Found negative signal: {signals[signals < 0].tolist()}"
        assert set(signals.unique()).issubset({0, 1})

    def test_no_short_signals_rising_price(self) -> None:
        """Strongly rising price (z >> z_entry) must clip to 0, not -1."""
        prices_list = list(range(50, 110))  # monotonically rising
        df = make_prices({"III": prices_list})
        strategy = MeanReversionStrategy(window=_WINDOW, z_entry=1.5)
        out = strategy.compute(df)

        signals = out[out["ticker"] == "III"]["signal"].dropna()
        assert (signals >= 0).all()


# ---------------------------------------------------------------------------
# TestOutputSchema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """Output DataFrame structure and dtypes."""

    def _run(self) -> pd.DataFrame:
        prices_list = [100.0 + i * 0.1 for i in range(30)]
        df = make_prices({"JJJ": prices_list})
        return MeanReversionStrategy(window=_WINDOW).compute(df)

    def test_output_columns(self) -> None:
        out = self._run()
        expected = {"ticker", "date", "close", "ma", "std", "z_score", "signal", "strategy"}
        assert set(out.columns) == expected

    def test_strategy_column_value(self) -> None:
        out = self._run()
        assert (out["strategy"] == "meanreversionstrategy").all()

    def test_ticker_column_populated(self) -> None:
        out = self._run()
        assert (out["ticker"] == "JJJ").all()

    def test_one_row_per_bar(self) -> None:
        prices_list = [100.0] * 30
        df = make_prices({"KKK": prices_list})
        out = MeanReversionStrategy(window=_WINDOW).compute(df)
        assert len(out) == 30

    def test_empty_prices_returns_empty_df(self) -> None:
        empty = pd.DataFrame(
            columns=["close"],
            index=pd.MultiIndex.from_tuples([], names=["ticker", "date"]),
        )
        out = MeanReversionStrategy(window=_WINDOW).compute(empty)
        assert out.empty
        assert set(out.columns) == {
            "ticker", "date", "close", "ma", "std", "z_score", "signal", "strategy"
        }

    def test_multi_ticker_row_count(self) -> None:
        df = make_prices({
            "T1": [100.0] * 25,
            "T2": [200.0] * 30,
        })
        out = MeanReversionStrategy(window=_WINDOW).compute(df)
        assert len(out[out["ticker"] == "T1"]) == 25
        assert len(out[out["ticker"] == "T2"]) == 30
