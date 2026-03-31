"""app.routers.signals — mean-reversion signal endpoint.

Requires ``X-API-Key`` header authentication.  Delegates all quantitative
work to :class:`~app.services.signal_engine.SignalEngine`.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from app.core.config import Settings, get_settings
from app.schemas.requests import SignalRequest
from app.schemas.responses import SignalResponse
from app.services.signal_engine import SignalEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Signals"])

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_TMP_DIR = Path("/tmp")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def _require_api_key(
    api_key: str | None = Security(_API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> None:
    """Raise HTTP 401 if the ``X-API-Key`` header is missing or invalid."""
    if api_key is None or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


def _save_signals_parquet(signals: SignalResponse) -> None:
    """Persist signal response to ``/tmp/signals_YYYYMMDD.parquet``."""
    today_str = date.today().strftime("%Y%m%d")
    path = _TMP_DIR / f"signals_{today_str}.parquet"
    try:
        rows = [s.model_dump() for s in signals.signals]
        pd.DataFrame(rows).to_parquet(path, index=False)
        logger.info("Background task — signals saved to %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Background task — failed to save parquet: %s", exc)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/signals",
    response_model=SignalResponse,
    summary="Run mean-reversion signal scan",
    dependencies=[Depends(_require_api_key)],
)
def run_signals(
    request: SignalRequest,
    background_tasks: BackgroundTasks,
) -> SignalResponse:
    """Execute a mean-reversion signal scan on the pre-built universe.

    Reads ``output/universe_*.parquet`` (built by ``equity-universe-scan``),
    selects the top-N value tickers, fetches OHLCV from Alpaca (with yfinance
    fallback), and returns Z-score signals.

    **Authentication**: ``X-API-Key`` header required.

    Raises:
        HTTP 401: Missing or invalid API key.
        HTTP 503: No universe file found — run ``equity-universe-scan`` first.
        HTTP 502: OHLCV fetch or signal computation failed.
    """
    try:
        engine = SignalEngine()
        result = engine.run_scan(request)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    background_tasks.add_task(_save_signals_parquet, result)
    return result
