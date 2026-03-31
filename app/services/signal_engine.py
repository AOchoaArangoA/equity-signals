"""app.services.signal_engine — computes mean-reversion signals.

Delegates to :class:`~equity_signals.data.alpaca_loader.AlpacaLoader` and
:class:`~equity_signals.strategies.mean_reversion.MeanReversionStrategy`.
Never imports UniverseFilter or TickerLoader — the caller decides which
tickers to analyse.
"""

from __future__ import annotations

import logging
import math
from datetime import date

import pandas as pd

from equity_signals.data.alpaca_loader import AlpacaLoader
from equity_signals.strategies.mean_reversion import MeanReversionStrategy

from app.schemas.requests import SignalRequest
from app.schemas.responses import SignalResponse, TickerSignal

logger = logging.getLogger(__name__)


class SignalEngine:
    """Computes mean-reversion signals for an explicit list of tickers.

    Example::

        engine = SignalEngine()
        response = engine.run(request)
    """

    def run(self, request: SignalRequest) -> SignalResponse:
        """Fetch OHLCV and compute signals for ``request.tickers``.

        Args:
            request: Validated :class:`~app.schemas.requests.SignalRequest`
                with a non-empty ``tickers`` list and strategy parameters.

        Returns:
            :class:`~app.schemas.responses.SignalResponse` with one row per
            ticker-date.

        Raises:
            RuntimeError: If OHLCV fetch or signal computation fails.
        """
        run_date = date.today().isoformat()
        logger.info(
            "SignalEngine.run — date=%s tickers=%s window=%d z_entry=%.2f days=%d",
            run_date, request.tickers, request.window, request.z_entry, request.days,
        )

        # ---- 1. Fetch OHLCV (Alpaca primary, yfinance fallback) --------
        try:
            prices = AlpacaLoader().get_ohlcv(request.tickers, days=request.days)
        except Exception as exc:
            raise RuntimeError(f"OHLCV fetch failed: {exc}") from exc

        if prices.empty:
            logger.warning("No price data returned — returning empty signals")
            return SignalResponse(
                run_date=run_date,
                ticker_count=0,
                signals=[],
            )

        # ---- 2. Compute signals ----------------------------------------
        try:
            strategy = MeanReversionStrategy(
                window=request.window,
                z_entry=request.z_entry,
                z_exit=request.z_exit,
            )
            signals_df = strategy.compute(prices)
        except Exception as exc:
            raise RuntimeError(f"Signal computation failed: {exc}") from exc

        # ---- 3. Build response -----------------------------------------
        signal_rows = _build_signal_rows(signals_df)
        ticker_count = signals_df["ticker"].nunique() if not signals_df.empty else 0

        logger.info(
            "SignalEngine.run complete — %d tickers, %d signal rows",
            ticker_count, len(signal_rows),
        )
        return SignalResponse(
            run_date=run_date,
            ticker_count=ticker_count,
            signals=signal_rows,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_signal_rows(df: pd.DataFrame) -> list[TickerSignal]:
    """Convert the signals DataFrame to :class:`TickerSignal` models.

    Rows where ``signal`` is NaN (warm-up period) are dropped.
    """
    rows: list[TickerSignal] = []
    for _, row in df.iterrows():
        if pd.isna(row.get("signal")):
            continue
        rows.append(
            TickerSignal(
                ticker=str(row["ticker"]),
                date=str(row["date"]),
                close=_safe_float(row.get("close")),
                ma=None if pd.isna(row.get("ma")) else float(row["ma"]),
                std=None if pd.isna(row.get("std")) else float(row["std"]),
                z_score=_safe_float(row.get("z_score")),
                signal=int(row["signal"]),
            )
        )
    return rows


def _safe_float(value: object) -> float:
    """Return a finite float, substituting 0.0 for NaN/None/inf."""
    try:
        f = float(value)  # type: ignore[arg-type]
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0
