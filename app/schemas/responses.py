"""app.schemas.responses — API response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TickerSignal(BaseModel):
    """Mean-reversion signal for a single ticker on a single date.

    Attributes:
        ticker: Uppercase ticker symbol.
        date: ISO-8601 date string (``YYYY-MM-DD``).
        close: Closing price.
        ma: Moving average value (``None`` for warm-up rows).
        std: Rolling standard deviation (``None`` for warm-up rows).
        z_score: Z-score of close relative to MA.
        signal: Long signal — ``1`` (enter long) or ``0`` (neutral/exit).
    """

    ticker: str
    date: str
    close: float
    ma: float | None = Field(default=None)
    std: float | None = Field(default=None)
    z_score: float
    signal: int = Field(ge=0, le=1)


class UniverseTicker(BaseModel):
    """Fundamental snapshot for a single ticker from the universe filter.

    Attributes:
        ticker: Uppercase ticker symbol.
        market_cap: Market capitalisation in USD.
        pb_ratio: Price-to-book ratio.
        roe: Return on equity.
        sector: GICS sector string.
        pb_rank_sector: Intra-sector P/B rank (1 = cheapest).
        value_signal: ``True`` if this ticker passed all filter stages.
    """

    ticker: str
    market_cap: float
    pb_ratio: float
    roe: float
    sector: str
    pb_rank_sector: int
    value_signal: bool


class UniverseResponse(BaseModel):
    """Response for ``POST /api/v1/universe``.

    Attributes:
        run_date: ISO-8601 date the scan was executed (``YYYY-MM-DD``).
        universe_size: Number of tickers with ``value_signal=True``.
        tickers: All tickers that passed the filter pipeline.
    """

    run_date: str
    universe_size: int
    tickers: list[UniverseTicker]


class Position(BaseModel):
    """Snapshot of a single open Alpaca position.

    Attributes:
        ticker: Uppercase ticker symbol.
        qty: Number of shares held.
        market_value: Current market value in USD.
        unrealized_pct: Unrealised P&L as a percentage.
    """

    ticker: str
    qty: float
    market_value: float
    unrealized_pct: float


class OrderConfirmation(BaseModel):
    """Confirmation of a submitted market order.

    Attributes:
        ticker: Uppercase ticker symbol.
        qty: Number of shares ordered.
        side: ``"buy"`` or ``"sell"``.
        status: Alpaca order status string (e.g. ``"accepted"``).
        order_id: Alpaca order UUID.
    """

    ticker: str
    qty: float
    side: str
    status: str
    order_id: str


class SignalResponse(BaseModel):
    """Response for ``POST /api/v1/signals``.

    Attributes:
        run_date: ISO-8601 date the scan was executed (``YYYY-MM-DD``).
        ticker_count: Number of tickers signals were computed for.
        signals: Mean-reversion signal rows (one per ticker-date).
    """

    run_date: str
    ticker_count: int
    signals: list[TickerSignal]
