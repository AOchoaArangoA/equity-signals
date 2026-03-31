"""mean_reversion — Z-score mean-reversion strategy.

:class:`MeanReversionStrategy` computes a long-only mean-reversion signal for
each ticker in a MultiIndex price DataFrame.  The signal fires when a stock's
closing price deviates significantly below its moving average, suggesting a
potential reversion upward.

Algorithm (per ticker)
-----------------------
1. Compute a rolling moving average (SMA or EMA) over *window* trading days.
2. Compute the rolling standard deviation of close prices over the same window.
3. Derive the Z-score: ``z = (close - MA) / std``.
4. Map Z-scores to signals:

   +--------------+---------------------------------------------------+
   | Condition    | Signal                                            |
   +==============+===================================================+
   | z < -z_entry | +1  (price well below MA → long entry)            |
   +--------------+---------------------------------------------------+
   | z > +z_entry | -1 before long-only clip (price well above MA)    |
   +--------------+---------------------------------------------------+
   | |z| ≤ z_exit | 0   (price near MA → neutral / exit)              |
   +--------------+---------------------------------------------------+

5. Long-only clip: all -1 values are mapped to 0.

All computations are fully vectorised — no ``iterrows``, no Python loops over
tickers.  Per-ticker errors are caught, logged, and returned as NaN rows.

Typical usage::

    from equity_signals.strategies.mean_reversion import MeanReversionStrategy

    strategy = MeanReversionStrategy(window=20, z_entry=1.5)
    signals = strategy.compute(prices_multiindex_df)
"""

from __future__ import annotations

import logging

import pandas as pd

from equity_signals.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Output columns produced by this strategy (in order).
_OUTPUT_COLS: list[str] = [
    "ticker", "date", "close", "ma", "std", "z_score", "signal", "strategy",
]


class MeanReversionStrategy(BaseStrategy):
    """Long-only Z-score mean-reversion signal generator.

    For each ticker the strategy computes a rolling moving average and
    standard deviation, derives a Z-score, and emits a binary long signal
    (1 = enter long, 0 = neutral / exit).

    Args:
        window: Lookback window in trading days for the MA and rolling std.
            Default ``20``.
        z_entry: Absolute Z-score threshold to trigger a signal.  A long
            signal fires when ``z < -z_entry``.  Default ``1.5``.
        z_exit: Absolute Z-score threshold to exit (return to neutral).
            When ``|z| <= z_exit`` the signal is forced to 0.  Default
            ``0.5``.
        use_ema: When ``True`` the moving average is an exponential moving
            average (EMA); otherwise a simple moving average (SMA) is used.
            Default ``False``.

    Example::

        strategy = MeanReversionStrategy(window=20, z_entry=1.5, z_exit=0.5)
        signals = strategy.compute(prices_df)
    """

    def __init__(
        self,
        *,
        window: int = 20,
        z_entry: float = 1.5,
        z_exit: float = 0.5,
        use_ema: bool = False,
    ) -> None:
        self._window = window
        self._z_entry = z_entry
        self._z_exit = z_exit
        self._use_ema = use_ema

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute mean-reversion signals for all tickers in *prices*.

        Processes every ticker independently using fully vectorised pandas
        operations via ``groupby``.  A ticker with fewer rows than *window*
        or that raises an unexpected error is returned as a NaN row with a
        warning — it never causes the whole batch to fail.

        Args:
            prices: MultiIndex DataFrame with levels ``(ticker, date)`` sorted
                ascending, containing at least a ``close`` column.

        Returns:
            Flat DataFrame with columns ``ticker, date, close, ma, std,
            z_score, signal, strategy``.  One row per ticker-date.  Tickers
            with insufficient data have NaN for all derived columns.
        """
        if prices.empty:
            logger.warning("MeanReversionStrategy.compute — received empty prices DataFrame")
            return pd.DataFrame(columns=_OUTPUT_COLS)

        tickers = prices.index.get_level_values("ticker").unique().tolist()
        logger.info(
            "MeanReversionStrategy — computing signals for %d ticker(s) "
            "(window=%d, z_entry=%.2f, z_exit=%.2f, use_ema=%s)",
            len(tickers),
            self._window,
            self._z_entry,
            self._z_exit,
            self._use_ema,
        )

        parts: list[pd.DataFrame] = []
        for ticker in tickers:
            part = self._compute_ticker(ticker, prices.xs(ticker, level="ticker"))
            parts.append(part)

        result = pd.concat(parts, ignore_index=True)

        # Log signal counts per ticker at INFO level.
        signal_counts = (
            result.groupby("ticker")["signal"]
            .value_counts()
            .rename("count")
            .reset_index()
        )
        for _, row in signal_counts.iterrows():
            logger.info(
                "  %s — signal=%d: %d bar(s)",
                row["ticker"],
                int(row["signal"]) if pd.notna(row["signal"]) else float("nan"),
                int(row["count"]),
            )

        return result[_OUTPUT_COLS]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_ticker(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """Compute signals for a single ticker.

        Args:
            ticker: Ticker symbol string.
            df: Single-level DataFrame indexed by ``date``, with a ``close``
                column.  (The ticker level has already been removed via
                ``xs``.)

        Returns:
            Flat DataFrame with :data:`_OUTPUT_COLS` columns and the ticker
            label filled in.
        """
        out = df[["close"]].copy()
        out.index.name = "date"
        n = len(out)

        if n < self._window:
            logger.warning(
                "MeanReversionStrategy — %s has %d rows, need at least %d "
                "(window=%d); returning NaN row",
                ticker, n, self._window, self._window,
            )
            nan_row = pd.DataFrame(
                [{
                    "ticker": ticker,
                    "date": out.index[-1] if n > 0 else pd.NaT,
                    "close": out["close"].iloc[-1] if n > 0 else float("nan"),
                    "ma": float("nan"),
                    "std": float("nan"),
                    "z_score": float("nan"),
                    "signal": float("nan"),
                    "strategy": self.name,
                }]
            )
            return nan_row

        try:
            # ---- moving average ----------------------------------------
            if self._use_ema:
                out["ma"] = (
                    out["close"]
                    .ewm(span=self._window, adjust=False, min_periods=self._window)
                    .mean()
                )
            else:
                out["ma"] = out["close"].rolling(window=self._window, min_periods=self._window).mean()

            # ---- rolling standard deviation ----------------------------
            out["std"] = out["close"].rolling(window=self._window, min_periods=self._window).std()

            # ---- Z-score -----------------------------------------------
            # Where std == 0 (flat price), set z_score = 0 to avoid divide-by-zero.
            out["z_score"] = (out["close"] - out["ma"]) / out["std"].replace(0, float("nan"))
            out["z_score"] = out["z_score"].fillna(0.0)

            # ---- signal (before long-only clip) ------------------------
            # Default to 0; overwrite for entry/exit zones.
            signal = pd.Series(0, index=out.index, dtype=int)
            signal = signal.where(out["z_score"].abs() <= self._z_exit, signal)   # neutral zone kept 0
            signal = signal.where(out["z_score"] >= -self._z_entry, signal)        # below entry → long
            signal[out["z_score"] < -self._z_entry] = 1
            signal[out["z_score"] > self._z_entry] = -1  # will be clipped below

            # ---- long-only clip ----------------------------------------
            out["signal"] = signal.clip(lower=0)

            out["strategy"] = self.name
            out = out.reset_index()  # date → column
            out.insert(0, "ticker", ticker)

            return out

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MeanReversionStrategy — unexpected error for %s (%s: %s); "
                "returning NaN row",
                ticker, type(exc).__name__, exc,
            )
            nan_row = pd.DataFrame(
                [{
                    "ticker": ticker,
                    "date": pd.NaT,
                    "close": float("nan"),
                    "ma": float("nan"),
                    "std": float("nan"),
                    "z_score": float("nan"),
                    "signal": float("nan"),
                    "strategy": self.name,
                }]
            )
            return nan_row
