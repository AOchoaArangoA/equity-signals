"""app.services.signal_engine — orchestrates the equity_signals pipeline.

:class:`SignalEngine` is the single point of contact between the FastAPI layer
and the quantitative package.  It:

1. Loads the latest universe snapshot via
   :func:`~equity_signals.universe.universe_store.load_latest_universe`.
2. Selects the top-N value tickers by ``pb_rank_sector``.
3. Fetches OHLCV history via
   :class:`~equity_signals.data.alpaca_loader.AlpacaLoader` (with automatic
   yfinance fallback).
4. Computes Z-score mean-reversion signals via
   :class:`~equity_signals.strategies.mean_reversion.MeanReversionStrategy`.
5. Returns a :class:`~app.schemas.responses.SignalResponse`.

No quantitative logic lives here — this class only wires existing components.
"""

from __future__ import annotations

import logging
import math
from datetime import date

import pandas as pd

from equity_signals.data.alpaca_loader import AlpacaLoader
from equity_signals.strategies.mean_reversion import MeanReversionStrategy
from equity_signals.universe.universe_store import load_latest_universe

from app.schemas.requests import SignalRequest
from app.schemas.responses import SignalResponse, TickerSignal, UniverseTicker

logger = logging.getLogger(__name__)


class SignalEngine:
    """Orchestrates the equity-signals pipeline for a single API request.

    Example::

        engine = SignalEngine()
        response = engine.run_scan(request)
    """

    def run_scan(self, request: SignalRequest) -> SignalResponse:
        """Execute the full signal scan and return a :class:`SignalResponse`.

        Args:
            request: Validated :class:`~app.schemas.requests.SignalRequest`
                containing filter parameters and strategy settings.

        Returns:
            :class:`~app.schemas.responses.SignalResponse` with universe
            summary and per-ticker signal rows.

        Raises:
            FileNotFoundError: If no universe parquet exists in ``output/``.
                Caller should return HTTP 503.
            RuntimeError: If OHLCV fetch or signal computation fails.
                Caller should return HTTP 502.
        """
        run_date = date.today().isoformat()
        logger.info(
            "SignalEngine.run_scan — date=%s top_n=%d window=%d z_entry=%.2f days=%d",
            run_date, request.top_n, request.window, request.z_entry, request.days,
        )

        # ---- 1. Load universe ------------------------------------------
        universe_df = load_latest_universe()
        logger.info("Universe loaded — %d rows", len(universe_df))

        # ---- 2. Select top-N value tickers -----------------------------
        value_df = universe_df[universe_df["value_signal"] == True]  # noqa: E712
        if value_df.empty:
            logger.warning("No tickers with value_signal=True — returning empty response")
            return SignalResponse(
                run_date=run_date,
                universe_size=0,
                top_n=request.top_n,
                universe=[],
                signals=[],
            )

        top_df = value_df.nsmallest(request.top_n, "pb_rank_sector")
        top_tickers: list[str] = top_df["ticker"].tolist()
        logger.info("Top %d tickers: %s", len(top_tickers), top_tickers)

        # ---- 3. Fetch OHLCV -------------------------------------------
        try:
            prices = AlpacaLoader().get_ohlcv(top_tickers, days=request.days)
        except Exception as exc:
            raise RuntimeError(f"OHLCV fetch failed: {exc}") from exc

        # ---- 4. Compute signals ----------------------------------------
        try:
            strategy = MeanReversionStrategy(
                window=request.window,
                z_entry=request.z_entry,
            )
            signals_df = strategy.compute(prices)
        except Exception as exc:
            raise RuntimeError(f"Signal computation failed: {exc}") from exc

        # ---- 5. Build response -----------------------------------------
        universe_rows = _build_universe(value_df)
        signal_rows = _build_signals(signals_df)

        return SignalResponse(
            run_date=run_date,
            universe_size=len(value_df),
            top_n=len(top_tickers),
            universe=universe_rows,
            signals=signal_rows,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_universe(df: pd.DataFrame) -> list[UniverseTicker]:
    """Convert the value-signal universe DataFrame to response models."""
    rows: list[UniverseTicker] = []
    for _, row in df.iterrows():
        rows.append(
            UniverseTicker(
                ticker=str(row["ticker"]),
                market_cap=_safe_float(row.get("market_cap")),
                pb_ratio=_safe_float(row.get("pb_ratio")),
                roe=_safe_float(row.get("roe")),
                sector=str(row.get("sector") or ""),
                pb_rank_sector=int(row.get("pb_rank_sector", 0)),
                value_signal=bool(row.get("value_signal", False)),
            )
        )
    return rows


def _build_signals(df: pd.DataFrame) -> list[TickerSignal]:
    """Convert the signals DataFrame to response models, dropping NaN rows."""
    rows: list[TickerSignal] = []
    for _, row in df.iterrows():
        # Skip warm-up rows where signal is NaN.
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
