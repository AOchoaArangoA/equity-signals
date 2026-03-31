"""fmp_loader — FMP stable API client for price and profile endpoints.

.. note:: **FMP Endpoint Tier Reference**

   The table below documents which FMP stable API endpoints work on each
   subscription tier.  Fundamentals fetching (P/B, ROE) has been moved to
   :mod:`equity_signals.data.yfinance_loader` which has no subscription
   requirement.

   +-------------------------------------+--------+---------+-----------+
   | Endpoint                            | Free   | Basic   | Premium   |
   +=====================================+========+=========+===========+
   | ``/stable/profile?symbol=T``        | ✅     | ✅      | ✅        |
   +-------------------------------------+--------+---------+-----------+
   | ``/stable/profile?symbol=T1,T2,...``| ✅     | ✅      | ✅        |
   +-------------------------------------+--------+---------+-----------+
   | ``/stable/key-metrics-ttm?symbol=T``| ✅     | ✅      | ✅        |
   +-------------------------------------+--------+---------+-----------+
   | ``/stable/key-metrics-ttm``         | ❌     | ❌      | ✅ only   |
   | ``?symbol=T1,T2,...`` (multi)       |        |         |           |
   +-------------------------------------+--------+---------+-----------+

   Multi-symbol ``/key-metrics-ttm`` returns **HTTP 402** on free/basic
   plans.  Use single-symbol calls or switch to yfinance (no key required).

:class:`FMPLoader` is kept for FMP endpoints that are free-tier compatible,
such as ``/profile`` (single or bulk) and any future price/OHLCV endpoints
that FMP exposes under the stable API.

Typical usage::

    from equity_signals.data.fmp_loader import FMPLoader

    loader = FMPLoader()
    profile = loader.get_profile("AAPL")
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from equity_signals.config import settings
from equity_signals.exceptions import (
    FMPDataError,
    FMPRateLimitError,
    FMPResponseError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL: str = "https://financialmodelingprep.com/stable"
_MAX_RETRIES: int = 3
_BACKOFF_BASE: float = 2.0  # seconds; delay = _BACKOFF_BASE ** attempt


# ---------------------------------------------------------------------------
# FMPLoader
# ---------------------------------------------------------------------------


class FMPLoader:
    """Client for FMP stable API endpoints compatible with the free/basic tier.

    Fundamentals (P/B, ROE, market cap) are **not** fetched here — use
    :class:`~equity_signals.data.yfinance_loader.YFinanceLoader` instead,
    which requires no API key.

    This class is reserved for FMP endpoints that remain accessible on the
    free tier, such as ``/profile`` (company overview, sector, exchange).

    Args:
        api_key: FMP API key. Defaults to ``settings.fmp_api_key``.
            May be ``None`` if only endpoints that do not require a key are
            used.
        base_url: Root URL of the FMP stable API. Override in tests.
        max_retries: Maximum retry attempts on rate-limit (429) responses.
        backoff_base: Exponential back-off base in seconds.

    Example::

        loader = FMPLoader()
        profile = loader.get_profile("AAPL")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _BASE_URL,
        max_retries: int = _MAX_RETRIES,
        backoff_base: float = _BACKOFF_BASE,
    ) -> None:
        self._api_key: str | None = api_key or settings.fmp_api_key
        self._base_url: str = base_url.rstrip("/")
        self._max_retries: int = max_retries
        self._backoff_base: float = backoff_base
        self._session: requests.Session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_profile(self, ticker: str) -> dict[str, Any]:
        """Return the company profile for *ticker* from ``/stable/profile``.

        Available on free and basic FMP plans.

        Args:
            ticker: Uppercase ticker symbol, e.g. ``"AAPL"``.

        Returns:
            Dict containing FMP profile fields such as ``mktCap``, ``sector``,
            ``industry``, ``exchange``, ``ipoDate``, etc.  Returns an empty
            dict if FMP returns no data for the ticker.

        Raises:
            :class:`~equity_signals.exceptions.FMPRateLimitError`:
                When rate-limit retries are exhausted.
            :class:`~equity_signals.exceptions.FMPResponseError`:
                On non-200, non-429 HTTP status codes.
        """
        results = self._get("/profile", ticker)
        return results[0] if results else {}

    # ------------------------------------------------------------------
    # Private — HTTP helper
    # ------------------------------------------------------------------

    def _get(self, path: str, ticker: str) -> list[dict[str, Any]]:
        """Execute a single-symbol GET request with exponential back-off on 429.

        Args:
            path: Endpoint path relative to :attr:`_base_url`,
                e.g. ``"/profile"``.
            ticker: Ticker symbol passed as ``?symbol=``.

        Returns:
            Parsed JSON list.  A single-object response is normalised to a
            one-element list.

        Raises:
            :class:`~equity_signals.exceptions.FMPRateLimitError`:
                When all retries are exhausted due to rate limiting.
            :class:`~equity_signals.exceptions.FMPResponseError`:
                On any non-200, non-429 HTTP status code.
            :class:`~equity_signals.exceptions.FMPDataError`:
                When the response body is neither a list nor a dict.
        """
        url = f"{self._base_url}{path}"
        params: dict[str, str] = {"symbol": ticker}
        if self._api_key:
            params["apikey"] = self._api_key

        for attempt in range(self._max_retries):
            logger.debug(
                "GET %s?symbol=%s (attempt %d/%d)",
                url, ticker, attempt + 1, self._max_retries,
            )
            response = self._session.get(url, params=params, timeout=15)

            if response.status_code == 429:
                if attempt == self._max_retries - 1:
                    raise FMPRateLimitError(ticker, self._max_retries)
                delay = self._backoff_base ** attempt
                logger.warning(
                    "Rate limited on %s — retrying in %.1fs (attempt %d/%d)",
                    ticker, delay, attempt + 1, self._max_retries,
                )
                time.sleep(delay)
                continue

            if response.status_code != 200:
                raise FMPResponseError(
                    ticker,
                    response.status_code,
                    response.text[:200],
                )

            data = response.json()
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                raise FMPDataError(
                    ticker,
                    f"expected a JSON list but got {type(data).__name__}",
                )
            return data

        raise FMPRateLimitError(ticker, self._max_retries)  # pragma: no cover
