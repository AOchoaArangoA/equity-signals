"""app.routers.signals — universe and signal endpoints.

Two independent, stateless endpoints:

* ``POST /api/v1/universe`` — builds the investable universe from Russell 2000
  fundamentals.  Never imports MeanReversionStrategy.

* ``POST /api/v1/signals`` — computes mean-reversion signals for any caller-
  supplied ticker list.  Never imports UniverseFilter.

Both require ``X-API-Key`` header authentication and return synchronously.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from app.core.config import Settings, get_settings
from app.schemas.requests import SignalRequest, UniverseRequest
from app.schemas.responses import SignalResponse, UniverseResponse
from app.services.signal_engine import SignalEngine
from app.services.universe_service import UniverseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Signals"])

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


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
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/universe",
    response_model=UniverseResponse,
    summary="Build investable universe",
    dependencies=[Depends(_require_api_key)],
)
def build_universe(request: UniverseRequest) -> UniverseResponse:
    """Generate the filtered investable universe from Russell 2000 fundamentals.

    Downloads the Russell 2000 constituent list from iShares IWM, fetches
    fundamental data via yfinance, and applies the four-stage filter pipeline
    (mid-cap → sector → ROE → intra-sector P/B ranking).

    Returns all tickers that passed the pipeline.  The caller can then select
    a subset and pass them to ``POST /api/v1/signals``.

    **Authentication**: ``X-API-Key`` header required.

    Raises:
        HTTP 401: Missing or invalid API key.
        HTTP 502: Ticker download or filter pipeline failed.
    """
    try:
        return UniverseService().run(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post(
    "/signals",
    response_model=SignalResponse,
    summary="Compute mean-reversion signals",
    dependencies=[Depends(_require_api_key)],
)
def compute_signals(request: SignalRequest) -> SignalResponse:
    """Compute Z-score mean-reversion signals for a caller-supplied ticker list.

    Fetches OHLCV history from Alpaca Markets (with automatic yfinance
    fallback) and applies the mean-reversion strategy.  The ticker list can
    come from anywhere — a previous ``/universe`` call, a watchlist, or any
    other source.

    **Authentication**: ``X-API-Key`` header required.

    Raises:
        HTTP 401: Missing or invalid API key.
        HTTP 502: OHLCV fetch or signal computation failed.
    """
    try:
        return SignalEngine().run(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
