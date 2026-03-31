"""alpaca_loader — market data loader backed by the Alpaca Markets API (alpaca-py).

Provides :class:`AlpacaLoader` which wraps the ``alpaca-py`` SDK to fetch
daily OHLCV bars and latest trade prices for a list of ticker symbols.

Design
------
* **Single batched call** — ``alpaca-py`` natively accepts a list of symbols
  in one request, so all tickers are fetched in a single API call.
* **Forward-fill** — missing trading days within each ticker are filled
  forward up to ``settings.alpaca_ffill_limit`` consecutive days using a
  vectorized ``groupby(level="ticker").ffill()`` to prevent cross-ticker
  contamination.
* **Observability** — logs total tickers, date range, and any tickers
  whose bar count is below the expected business-day count for the window.

Paper and live environments are handled transparently via ``ALPACA_BASE_URL``.

Typical usage::

    from equity_signals.data.alpaca_loader import AlpacaLoader

    loader = AlpacaLoader()
    ohlcv = loader.get_ohlcv(["AAPL", "MSFT", "GOOG"])
    prices = loader.get_latest_prices(["AAPL", "MSFT"])
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

from equity_signals.config import settings
from equity_signals.data.yfinance_loader import fetch_ohlcv
from equity_signals.exceptions import AlpacaAPIError, AlpacaDataError

logger = logging.getLogger(__name__)

# Substring that identifies the paper-trading base URL.
_PAPER_MARKER: str = "paper"


class AlpacaLoader:
    """Fetches market data from Alpaca using the ``alpaca-py`` SDK.

    Credentials and the target environment (paper vs live) are resolved from
    :data:`equity_signals.config.settings` at construction time.

    Args:
        api_key: Alpaca API key. Defaults to ``settings.alpaca_api_key``.
        secret_key: Alpaca secret key. Defaults to ``settings.alpaca_secret_key``.
        base_url: Trading base URL. Used to detect paper vs live environment.
        url_override: When set, passed to ``StockHistoricalDataClient`` as
            ``url_override`` to redirect all data requests (useful for tests).
        ffill_limit: Maximum consecutive missing trading days to forward-fill.
            Defaults to ``settings.alpaca_ffill_limit``.

    Attributes:
        is_paper: ``True`` when *base_url* contains ``"paper"``.

    Example::

        loader = AlpacaLoader()
        df = loader.get_ohlcv(["AAPL", "TSLA"], days=30)
        prices = loader.get_latest_prices(["AAPL", "TSLA"])
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        url_override: str | None = None,
        ffill_limit: int | None = None,
    ) -> None:
        self._api_key: str = api_key or settings.alpaca_api_key
        self._secret_key: str = secret_key or settings.alpaca_secret_key
        self._base_url: str = base_url or settings.alpaca_base_url
        self._ffill_limit: int = ffill_limit if ffill_limit is not None else settings.alpaca_ffill_limit

        self.is_paper: bool = _PAPER_MARKER in self._base_url.lower()

        logger.debug(
            "AlpacaLoader initialised — environment: %s, ffill_limit: %d",
            "paper" if self.is_paper else "live",
            self._ffill_limit,
        )

        # Single API client; data endpoint is the same for paper and live.
        self._client: StockHistoricalDataClient = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            url_override=url_override,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        tickers: list[str],
        days: int = 90,
    ) -> pd.DataFrame:
        """Return daily OHLCV bars for *tickers* over the last *days* calendar days.

        **Primary source**: Alpaca Markets API (single batched call).
        **Fallbacks** (applied automatically, logged at WARNING):

        1. If Alpaca raises any exception → all tickers fetched via yfinance.
        2. If Alpaca returns data but some tickers are missing from the result →
           the missing subset is fetched via yfinance and merged.

        The method never raises — if all sources fail for a ticker it is simply
        absent from the returned DataFrame.

        Args:
            tickers: Uppercase ticker symbols, e.g. ``["AAPL", "MSFT"]``.
            days: Number of calendar days to look back from today. Default 90.

        Returns:
            DataFrame with MultiIndex ``(ticker, date)`` sorted ascending, with
            columns ``open, high, low, close, volume``.  Empty (but correctly
            structured) if no data could be fetched from any source.
        """
        end: datetime = datetime.now(tz=timezone.utc)
        start: datetime = end - timedelta(days=days)

        logger.info(
            "AlpacaLoader.get_ohlcv — %d tickers, %s to %s",
            len(tickers),
            start.date(),
            end.date(),
        )

        # ---- primary: Alpaca --------------------------------------------
        try:
            alpaca_df = self._fetch_alpaca(tickers, start, end)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Alpaca OHLCV fetch failed for %d tickers: %s — falling back to yfinance",
                len(tickers),
                exc,
            )
            return fetch_ohlcv(tickers, days)

        # ---- partial fallback: tickers missing from Alpaca response -----
        returned = set(alpaca_df.index.get_level_values("ticker").unique())
        missing = set(tickers) - returned
        if missing:
            logger.warning(
                "Alpaca missing %d ticker(s): %s — fetching via yfinance",
                len(missing),
                sorted(missing),
            )
            yf_df = fetch_ohlcv(list(missing), days)
            if not yf_df.empty:
                alpaca_df = pd.concat([alpaca_df, yf_df]).sort_index()

        return alpaca_df

    # ------------------------------------------------------------------
    # Private — Alpaca fetch
    # ------------------------------------------------------------------

    def _fetch_alpaca(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch bars from Alpaca and return a normalised MultiIndex DataFrame.

        Args:
            tickers: Ticker symbols to fetch.
            start: UTC start datetime.
            end: UTC end datetime.

        Returns:
            MultiIndex DataFrame ``(ticker, date)`` with lowercase OHLCV columns.

        Raises:
            :class:`~equity_signals.exceptions.AlpacaAPIError`:
                When the Alpaca API returns an error response.
            :class:`~equity_signals.exceptions.AlpacaDataError`:
                When the response contains no bar data.
        """
        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        raw: Any = self._call(lambda: self._client.get_stock_bars(request), tickers)
        df: pd.DataFrame = raw.df

        if df.empty:
            raise AlpacaDataError(
                tickers, f"no bars returned for the last {(end - start).days} days"
            )

        # Normalise MultiIndex (symbol, timestamp) → (ticker, date).
        df = df.rename_axis(index={"symbol": "ticker", "timestamp": "date"})
        ticker_level = df.index.get_level_values("ticker")
        date_level = df.index.get_level_values("date").date
        df.index = pd.MultiIndex.from_arrays(
            [ticker_level, date_level],
            names=["ticker", "date"],
        )
        df = df.sort_index()

        # Observability: flag tickers with fewer bars than expected.
        expected_days = len(pd.bdate_range(start.date(), end.date()))
        bars_per_ticker = df.groupby(level="ticker").size()
        for tkr, count in bars_per_ticker[bars_per_ticker < expected_days].items():
            logger.warning(
                "Ticker %s has %d/%d expected trading days — gaps will be forward-filled",
                tkr, count, expected_days,
            )

        # Vectorised forward-fill within each ticker (no cross-ticker bleed).
        df = df.groupby(level="ticker").ffill(limit=self._ffill_limit)

        logger.info(
            "AlpacaLoader._fetch_alpaca — %d rows across %d ticker(s)",
            len(df),
            bars_per_ticker.shape[0],
        )
        return df

    def get_latest_prices(self, tickers: list[str]) -> pd.Series:
        """Return the latest trade price for each ticker in *tickers*.

        Args:
            tickers: Uppercase ticker symbols, e.g. ``["AAPL", "MSFT"]``.

        Returns:
            :class:`pandas.Series` indexed by ticker (``index.name="ticker"``)
            with the latest trade price as float values and ``name="price"``.

        Raises:
            :class:`~equity_signals.exceptions.AlpacaAPIError`:
                When the Alpaca API returns an error response.
            :class:`~equity_signals.exceptions.AlpacaDataError`:
                When the response contains no trade data.
        """
        logger.info("AlpacaLoader.get_latest_prices — %d ticker(s)", len(tickers))

        request = StockLatestTradeRequest(symbol_or_symbols=tickers)
        trades: dict[str, Any] = self._call(
            lambda: self._client.get_stock_latest_trade(request), tickers
        )

        if not trades:
            raise AlpacaDataError(tickers, "latest trade response was empty")

        prices = pd.Series(
            {symbol: trade.price for symbol, trade in trades.items()},
            name="price",
            dtype=float,
        )
        prices.index.name = "ticker"

        logger.info(
            "AlpacaLoader.get_latest_prices — prices received for %d ticker(s)",
            len(prices),
        )
        return prices

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call(self, fn: Any, tickers: list[str]) -> Any:
        """Execute *fn* and translate SDK exceptions into package exceptions.

        Args:
            fn: Zero-argument callable that performs an alpaca-py SDK call.
            tickers: Tickers associated with the call (used in error messages).

        Returns:
            Whatever *fn* returns on success.

        Raises:
            :class:`~equity_signals.exceptions.AlpacaAPIError`:
                On any :class:`alpaca.common.exceptions.APIError`.
        """
        try:
            return fn()
        except APIError as exc:
            status = getattr(exc, "status_code", 0)
            message = str(exc)
            logger.error(
                "Alpaca API error for %s — HTTP %d: %s",
                tickers,
                status,
                message,
            )
            raise AlpacaAPIError(tickers, status, message) from exc
