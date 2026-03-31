"""app.routers.signals — mean-reversion signal endpoint.

Requires ``X-API-Key`` header authentication.  Delegates all quantitative
work to :class:`~app.services.signal_engine.SignalEngine`.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, Security
from fastapi.security.api_key import APIKeyHeader

from app.core.config import Settings, get_settings
from app.schemas.requests import SignalRequest
from app.schemas.responses import SignalResponse
from app.services.signal_engine import SignalEngine
from equity_signals.scripts.run_universe_scan import run as _run_universe_scan

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


def _run_universe_scan_bg() -> None:
    """Run the full universe scan and save output to ``/tmp/``."""
    today_str = date.today().strftime("%Y%m%d")
    path = _TMP_DIR / f"universe_{today_str}.parquet"
    try:
        df = _run_universe_scan()
        df.to_parquet(path, index=False)
        logger.info("Universe scan complete — %d rows saved to %s", len(df), path)
    except SystemExit:
        # run_universe_scan calls sys.exit(1) on pipeline errors; treat as failure.
        logger.error("Universe scan failed — check logs above for details")
    except Exception as exc:  # noqa: BLE001
        logger.error("Universe scan background task error: %s", exc, exc_info=True)


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
    "/universe/scan",
    status_code=202,
    summary="Trigger universe scan (async)",
    dependencies=[Depends(_require_api_key)],
)
def trigger_universe_scan(
    background_tasks: BackgroundTasks,
) -> Response:
    """Enqueue a full universe scan and return immediately (HTTP 202).

    The scan runs in the background: downloads the Russell 2000, fetches
    fundamentals via yfinance, applies UniverseFilter, and saves the result
    to ``/tmp/universe_YYYYMMDD.parquet``.

    Once complete, ``POST /api/v1/signals`` will automatically pick up the
    new file via :func:`~equity_signals.universe.universe_store.load_latest_universe`.

    **Authentication**: ``X-API-Key`` header required.

    Returns:
        HTTP 202 with JSON body ``{"status": "accepted", "message": "..."}``.
    """
    today_str = date.today().strftime("%Y%m%d")
    output_path = str(_TMP_DIR / f"universe_{today_str}.parquet")

    background_tasks.add_task(_run_universe_scan_bg)
    logger.info("Universe scan enqueued — output will be written to %s", output_path)

    import json
    return Response(
        content=json.dumps({
            "status": "accepted",
            "message": "Universe scan started in background.",
            "path": output_path,
        }),
        status_code=202,
        media_type="application/json",
    )


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
