"""app.routers.orders — position and order management endpoints (stub).

Both endpoints delegate entirely to
:class:`~equity_signals.execution.AlpacaTrader` — no logic lives here.

Endpoints
---------
* ``GET  /api/v1/positions``          — list all open positions.
* ``POST /api/v1/orders/exit/{ticker}`` — submit a market-sell for *ticker*.

Both require ``X-API-Key`` header authentication.

.. note::
    ``AlpacaTrader`` is initialised with ``paper=True`` (hardcoded in the
    execution layer).  Live trading is not enabled.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Path, Security
from fastapi.security.api_key import APIKeyHeader

from app.core.config import Settings, get_settings
from app.schemas.responses import OrderConfirmation, Position
from equity_signals.execution import AlpacaTrader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Orders"])

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


@router.get(
    "/positions",
    response_model=list[Position],
    summary="List open positions",
    dependencies=[Depends(_require_api_key)],
)
def get_positions() -> list[Position]:
    """Return all currently open Alpaca paper-trading positions.

    Delegates to :meth:`~equity_signals.execution.AlpacaTrader.get_open_positions`.

    **Authentication**: ``X-API-Key`` header required.

    Raises:
        HTTP 401: Missing or invalid API key.
        HTTP 502: Alpaca API call failed.
    """
    try:
        trader = AlpacaTrader()
        positions = trader.get_open_positions()
    except Exception as exc:
        logger.error("get_positions failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Alpaca API error: {exc}") from exc

    return [
        Position(
            ticker=p["ticker"],
            qty=p["qty"],
            market_value=p["market_value"],
            unrealized_pct=p["unrealized_pct"],
        )
        for p in positions
    ]


@router.post(
    "/orders/exit/{ticker}",
    response_model=OrderConfirmation,
    summary="Submit market-sell for a position",
    dependencies=[Depends(_require_api_key)],
)
def exit_position(
    ticker: str = Path(
        ...,
        description="Ticker symbol to exit (uppercase, e.g. 'AAPL').",
        pattern=r"^[A-Z]{1,5}$",
    ),
) -> OrderConfirmation:
    """Submit a market-sell order for the full open position in *ticker*.

    Looks up the current held quantity from Alpaca, then submits a market-
    sell for that exact quantity.

    Delegates to :meth:`~equity_signals.execution.AlpacaTrader.get_open_positions`
    and :meth:`~equity_signals.execution.AlpacaTrader.submit_market_sell`.

    **Authentication**: ``X-API-Key`` header required.

    Raises:
        HTTP 401: Missing or invalid API key.
        HTTP 404: No open position found for *ticker*.
        HTTP 502: Alpaca API call failed.
    """
    try:
        trader = AlpacaTrader()
        positions = trader.get_open_positions()
    except Exception as exc:
        logger.error("exit_position — failed to fetch positions: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Alpaca API error: {exc}") from exc

    position = next((p for p in positions if p["ticker"] == ticker.upper()), None)
    if position is None:
        raise HTTPException(
            status_code=404,
            detail=f"No open position found for {ticker}.",
        )

    try:
        result = trader.submit_market_sell(ticker.upper(), int(position["qty"]))
    except Exception as exc:
        logger.error("exit_position — order failed for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Order submission failed: {exc}") from exc

    return OrderConfirmation(
        ticker=ticker.upper(),
        qty=position["qty"],
        side="sell",
        status=result["status"],
        order_id=result["order_id"],
    )
