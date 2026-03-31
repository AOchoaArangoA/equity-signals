"""app.services.universe_service — builds the investable universe.

Delegates to :class:`~equity_signals.universe.ticker_loader.TickerLoader` and
:class:`~equity_signals.universe.universe_filter.UniverseFilter`.
Never imports MeanReversionStrategy or AlpacaLoader.
"""

from __future__ import annotations

import logging
import math
from datetime import date

import pandas as pd

from pathlib import Path

from equity_signals.data.yfinance_loader import YFinanceLoader
from equity_signals.universe.ticker_loader import TickerLoader
from equity_signals.universe.universe_filter import FilterConfig, UniverseFilter

_TMP = Path("/tmp")

from app.schemas.requests import UniverseRequest
from app.schemas.responses import UniverseResponse, UniverseTicker

logger = logging.getLogger(__name__)


class UniverseService:
    """Builds the investable universe for a single API request.

    Example::

        service = UniverseService()
        response = service.run(request)
    """

    def run(self, request: UniverseRequest) -> UniverseResponse:
        """Run the universe filter pipeline and return a :class:`UniverseResponse`.

        Args:
            request: Validated :class:`~app.schemas.requests.UniverseRequest`.

        Returns:
            :class:`~app.schemas.responses.UniverseResponse` with all tickers
            that passed the filter pipeline.

        Raises:
            RuntimeError: If ticker download or filter pipeline fails.
        """
        run_date = date.today().isoformat()
        logger.info(
            "UniverseService.run — date=%s index_top_pct=%.0f%% pb_percentile=%.0f%%",
            run_date, request.index_top_pct, request.pb_percentile,
        )

        # ---- 1. Load tickers -------------------------------------------
        # Use /tmp for all cache I/O so the API process (non-root, read-only
        # working directory) never tries to write to .cache/ on disk.
        try:
            loader = TickerLoader(cache_dir=_TMP)
            tickers = (
                loader.get_top_pct(request.index_top_pct)
                if request.index_top_pct < 100.0
                else loader.get_russell2000()
            )
        except Exception as exc:
            raise RuntimeError(f"Ticker download failed: {exc}") from exc

        logger.info("%d tickers loaded", len(tickers))

        # ---- 2. Apply filters ------------------------------------------
        config = FilterConfig(
            midcap_min=request.midcap_min,
            midcap_max=request.midcap_max,
            sectors=request.sectors,
            pb_percentile=int(request.pb_percentile),
        )
        try:
            df = UniverseFilter(config, loader=YFinanceLoader(cache_dir=_TMP)).run(tickers)
        except Exception as exc:
            raise RuntimeError(f"Universe filter failed: {exc}") from exc

        # ---- 3. Build response -----------------------------------------
        universe_rows = _build_universe_tickers(df)
        value_count = int(df["value_signal"].sum()) if not df.empty else 0

        logger.info(
            "UniverseService.run complete — %d tickers, %d with value_signal=True",
            len(df), value_count,
        )
        return UniverseResponse(
            run_date=run_date,
            universe_size=value_count,
            tickers=universe_rows,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_universe_tickers(df: pd.DataFrame) -> list[UniverseTicker]:
    """Convert the filtered universe DataFrame to :class:`UniverseTicker` models."""
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


def _safe_float(value: object) -> float:
    """Return a finite float, substituting 0.0 for NaN/None/inf."""
    try:
        f = float(value)  # type: ignore[arg-type]
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0
