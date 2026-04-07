"""yfinance_loader — fundamental data loader backed by the yfinance library.

Fetches market capitalisation, price-to-book ratio, return-on-equity, and
sector for a list of ticker symbols using ``yfinance.Ticker(t).info``.

Design
------
* **Batched sequential fetch** — tickers are processed in batches of
  :data:`_BATCH_SIZE` (25).  Within each batch a small
  :class:`~concurrent.futures.ThreadPoolExecutor` (``max_workers=2``) is
  used.  A configurable inter-batch sleep prevents Yahoo Finance from rate-
  limiting the caller.
* **Per-request jitter** — each ``_fetch_ticker`` call sleeps
  ``uniform(0.5, 1.5)`` seconds before hitting the API, spreading load
  across the batch window.
* **Retry with exponential back-off** — up to :data:`_MAX_RETRIES` attempts
  per ticker (2 s → 4 s → 8 s).  Auth-related failures (HTTP 401 / invalid
  crumb) additionally trigger a session refresh before the next attempt.
* **No custom session** — yfinance 1.2+ manages its own ``curl_cffi``
  session internally.  Passing a ``requests.Session`` raises
  ``YFDataException`` in modern versions, so sessions are no longer injected.
* **Disk cache** — results are written to
  ``.cache/yf_fundamentals_YYYYMMDD.parquet``.  On subsequent calls within
  ``cache_ttl_days`` the cache is returned immediately, skipping all API
  calls.
* **Observability** — logs cache hit/miss, per-batch progress, missing
  fields, and total runtime.

``yfinance.Ticker.info`` field mapping
---------------------------------------

+--------------------+----------------------------+
| Output column      | ``info`` key               |
+====================+============================+
| ``market_cap``     | ``marketCap``              |
+--------------------+----------------------------+
| ``pb_ratio``       | ``priceToBook``            |
+--------------------+----------------------------+
| ``roe``            | ``returnOnEquity``         |
+--------------------+----------------------------+
| ``sector``         | ``sector``                 |
+--------------------+----------------------------+

Typical usage::

    from equity_signals.data.yfinance_loader import YFinanceLoader

    loader = YFinanceLoader()
    df = loader.get_fundamentals(["AAPL", "MSFT", "GOOG"])
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from equity_signals.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR: Path = Path(".cache")
_CACHE_GLOB: str = "yf_fundamentals_*.parquet"
_CACHE_PREFIX: str = "yf_fundamentals_"

_FUNDAMENTALS_COLS: list[str] = [
    "ticker", "market_cap", "pb_ratio", "roe", "sector", "updated_at"
]

# yfinance .info key → output column name
_INFO_FIELD_MAP: dict[str, str] = {
    "marketCap": "market_cap",
    "priceToBook": "pb_ratio",
    "returnOnEquity": "roe",
    "sector": "sector",
}

_BATCH_SIZE: int = 25
_DEFAULT_MAX_WORKERS: int = 2
_MAX_RETRIES: int = 3
_RETRY_BACKOFF: tuple[float, ...] = (2.0, 4.0, 8.0)  # seconds per attempt

_AUTH_ERROR_SIGNALS: tuple[str, ...] = ("401", "crumb", "unauthorized", "invalid cookie")


# ---------------------------------------------------------------------------
# YFinanceLoader
# ---------------------------------------------------------------------------


class YFinanceLoader:
    """Fetches fundamental equity data via the ``yfinance`` library.

    Provides the same ``get_fundamentals`` interface as ``FMPLoader`` so it can
    be used as a drop-in replacement in
    :class:`~equity_signals.universe.universe_filter.UniverseFilter`.

    Args:
        max_workers: Concurrent threads *per batch*.  Defaults to 2.
            Keeping this low avoids Yahoo Finance rate limits when processing
            large universes.
        cache_ttl_days: Days before the parquet cache is considered stale.
            Defaults to ``settings.yfinance_cache_ttl_days``.
        cache_dir: Directory for parquet cache files. Defaults to ``.cache/``.

    Example::

        loader = YFinanceLoader()
        df = loader.get_fundamentals(["AAPL", "TSLA", "MSFT"])
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        cache_ttl_days: int | None = None,
        cache_dir: Path = _CACHE_DIR,
    ) -> None:
        self._max_workers: int = (
            max_workers if max_workers is not None else _DEFAULT_MAX_WORKERS
        )
        self._cache_ttl_days: int = (
            cache_ttl_days if cache_ttl_days is not None
            else settings.yfinance_cache_ttl_days
        )
        self._cache_dir: Path = cache_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """Return fundamental data for *tickers* as a :class:`pandas.DataFrame`.

        Serves from the parquet cache when a fresh file is available —
        skipping every API call.  Otherwise fetches tickers in sequential
        batches of :data:`_BATCH_SIZE` with a small thread pool per batch and
        inter-batch sleep to stay within Yahoo Finance's free-tier limits.

        A ticker that raises any exception during fetching is not dropped —
        it is returned as a row with ``NaN`` for all numeric fields so
        downstream filters can decide what to do with it.

        Args:
            tickers: List of uppercase ticker symbols, e.g. ``["AAPL", "MSFT"]``.

        Returns:
            DataFrame with columns:

            +------------+---------------------------------------------------+
            | Column     | Description                                       |
            +============+===================================================+
            | ticker     | Ticker symbol (str)                               |
            +------------+---------------------------------------------------+
            | market_cap | Market capitalisation in USD (float or NaN)       |
            +------------+---------------------------------------------------+
            | pb_ratio   | Price-to-book ratio (float or NaN)                |
            +------------+---------------------------------------------------+
            | roe        | Return on equity (float or NaN)                   |
            +------------+---------------------------------------------------+
            | sector     | GICS sector string (str or None)                  |
            +------------+---------------------------------------------------+
            | updated_at | UTC timestamp when the row was fetched            |
            +------------+---------------------------------------------------+
        """
        # ---- cache check -----------------------------------------------
        cached = self._load_cache()
        if cached is not None:
            cached_set = set(cached["ticker"].tolist())
            requested_set = set(tickers)
            # Accept the cache only when it covers ≥ 80 % of requested tickers.
            # This prevents a small per-watchlist cache (e.g. 2 rows) from
            # being returned for a large universe request (e.g. 1 900 tickers).
            coverage = len(requested_set & cached_set) / max(len(requested_set), 1)
            if coverage >= 0.80:
                logger.info(
                    "Cache hit — %d/%d requested tickers covered (%.0f%%); "
                    "skipping API calls (cache age < %d days)",
                    len(requested_set & cached_set),
                    len(requested_set),
                    coverage * 100,
                    self._cache_ttl_days,
                )
                # Return only the rows the caller asked for.
                return (
                    cached[cached["ticker"].isin(requested_set)]
                    .reset_index(drop=True)
                )
            else:
                logger.info(
                    "Cache miss — cached %d rows cover only %.0f%% of %d requested "
                    "tickers; fetching fresh data",
                    len(cached),
                    coverage * 100,
                    len(requested_set),
                )

        if not tickers:
            return pd.DataFrame(columns=_FUNDAMENTALS_COLS)

        # ---- sequential batched fetch ----------------------------------
        t0 = time.perf_counter()
        total = len(tickers)
        logger.info(
            "YFinanceLoader — fetching %d tickers in batches of %d (max_workers=%d)",
            total,
            _BATCH_SIZE,
            self._max_workers,
        )

        rows: list[dict[str, Any]] = []

        for batch_start in range(0, total, _BATCH_SIZE):
            batch = tickers[batch_start : batch_start + _BATCH_SIZE]

            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {pool.submit(self._fetch_ticker, t): t for t in batch}
                for future in as_completed(futures):
                    rows.append(future.result())  # never raises — errors become NaN rows

            processed = min(batch_start + _BATCH_SIZE, total)
            logger.info("Processed %d/%d tickers", processed, total)

            if processed < total:
                sleep_secs = random.uniform(2.0, 4.0)
                logger.debug("Inter-batch sleep %.1fs before next batch", sleep_secs)
                time.sleep(sleep_secs)

        df = pd.DataFrame(rows)[_FUNDAMENTALS_COLS]

        # ---- observability: log tickers with missing pb_ratio -----------
        missing_pb = df[df["pb_ratio"].isna()]["ticker"].tolist()
        if missing_pb:
            logger.warning(
                "YFinanceLoader — %d ticker(s) have missing pb_ratio: %s%s",
                len(missing_pb),
                ", ".join(missing_pb[:10]),
                " ..." if len(missing_pb) > 10 else "",
            )

        nan_market_cap = int(df["market_cap"].isna().sum())
        elapsed = time.perf_counter() - t0
        logger.info(
            "Fundamentals fetched: %d tickers, %d returned NaN market_cap",
            len(df),
            nan_market_cap,
        )
        logger.info("YFinanceLoader — fetched %d rows in %.1fs", len(df), elapsed)

        self._save_cache(df)
        return df

    # ------------------------------------------------------------------
    # Private — single-ticker fetch with retry
    # ------------------------------------------------------------------

    def _fetch_ticker(self, ticker: str) -> dict[str, Any]:
        """Fetch fundamental data for a single *ticker* via ``yfinance``.

        Retries up to :data:`_MAX_RETRIES` times with exponential back-off.
        Auth failures (HTTP 401 / crumb expiry) additionally trigger a session
        refresh before the next attempt.

        Returns a dict with all :data:`_FUNDAMENTALS_COLS` keys — numeric
        fields are ``None`` when data is unavailable.  Never raises.

        Args:
            ticker: Uppercase ticker symbol.
        """
        row: dict[str, Any] = {
            "ticker": ticker,
            "market_cap": None,
            "pb_ratio": None,
            "roe": None,
            "sector": None,
            "updated_at": datetime.now(tz=timezone.utc),
        }

        for attempt in range(_MAX_RETRIES):
            # Per-request jitter spreads Yahoo Finance load
            time.sleep(random.uniform(0.5, 1.5))

            try:
                info: dict[str, Any] = yf.Ticker(ticker).info

                for info_key, col in _INFO_FIELD_MAP.items():
                    value = info.get(info_key)
                    if value == "N/A" or value == "":
                        value = None
                    row[col] = value

                missing = [
                    col for info_key, col in _INFO_FIELD_MAP.items()
                    if row[col] is None
                ]
                if missing:
                    logger.debug("Ticker %s — missing fields: %s", ticker, missing)

                return row  # success — exit retry loop

            except Exception as exc:  # noqa: BLE001
                if attempt < _MAX_RETRIES - 1:
                    backoff = _RETRY_BACKOFF[attempt]
                    logger.warning(
                        "YFinanceLoader — %s fetch failed (attempt %d/%d, %s: %s); "
                        "retrying in %.0fs",
                        ticker,
                        attempt + 1,
                        _MAX_RETRIES,
                        type(exc).__name__,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.warning(
                        "YFinanceLoader — failed to fetch %s after %d attempts "
                        "(%s: %s); returning NaN row",
                        ticker,
                        _MAX_RETRIES,
                        type(exc).__name__,
                        exc,
                    )

        return row

    # ------------------------------------------------------------------
    # Private — cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, for_date: date) -> Path:
        """Return the parquet path for *for_date*."""
        return self._cache_dir / f"{_CACHE_PREFIX}{for_date.strftime('%Y%m%d')}.parquet"

    def _load_cache(self) -> pd.DataFrame | None:
        """Return the most complete fresh cached DataFrame, else ``None``.

        When multiple parquet files exist within the TTL window, the one with
        the most rows is returned — not necessarily the most recent.  This
        prevents a small per-watchlist fetch (e.g. 2 tickers) from shadowing a
        large full-universe cache (e.g. 1 900 tickers) just because it was
        written later.
        """
        if not self._cache_dir.exists():
            return None

        today = date.today()
        fresh: list[Path] = []

        for path in self._cache_dir.glob(_CACHE_GLOB):
            date_str = path.stem[len(_CACHE_PREFIX):]
            try:
                file_date = datetime.strptime(date_str, "%Y%m%d").date()
            except ValueError:
                logger.debug("Ignoring unrecognised cache file: %s", path)
                continue
            age_days = (today - file_date).days
            if age_days < self._cache_ttl_days:
                fresh.append(path)

        if not fresh:
            return None

        # Pick the file with the most rows among all fresh candidates.
        best_path: Path | None = None
        best_rows: int = -1
        for path in fresh:
            try:
                import pyarrow.parquet as pq  # lazy import — pyarrow ships with pandas
                n_rows = pq.read_metadata(path).num_rows
            except Exception:
                n_rows = path.stat().st_size  # fall back to file size as a proxy
            if n_rows > best_rows:
                best_rows = n_rows
                best_path = path

        if best_path is None:
            return None

        logger.debug("Loading YFinance cache from %s (%d rows)", best_path.name, best_rows)
        return pd.read_parquet(best_path)

    def _save_cache(self, df: pd.DataFrame) -> None:
        """Write *df* to a dated parquet file, skipping if a larger fresh cache exists.

        A small per-watchlist fetch should not overwrite (or shadow) a large
        full-universe cache that is still within the TTL window.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Check whether any fresh file already has more rows than the new df.
        today = date.today()
        for path in self._cache_dir.glob(_CACHE_GLOB):
            date_str = path.stem[len(_CACHE_PREFIX):]
            try:
                file_date = datetime.strptime(date_str, "%Y%m%d").date()
            except ValueError:
                continue
            if (today - file_date).days >= self._cache_ttl_days:
                continue
            try:
                import pyarrow.parquet as pq
                existing_rows = pq.read_metadata(path).num_rows
            except Exception:
                existing_rows = 0
            if existing_rows > len(df):
                logger.debug(
                    "Skipping cache write — existing %s has %d rows vs %d new rows",
                    path.name,
                    existing_rows,
                    len(df),
                )
                return

        path = self._cache_path(today)
        df.to_parquet(path, index=False)
        logger.debug("YFinance cache written to %s (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# Module-level OHLCV helper (used by AlpacaLoader fallback)
# ---------------------------------------------------------------------------


def fetch_ohlcv(tickers: list[str], days: int = 90) -> pd.DataFrame:
    """Fetch daily OHLCV bars for *tickers* via ``yf.download``.

    Returns a MultiIndex DataFrame with levels ``(ticker, date)`` and
    lowercase columns ``open, high, low, close, volume`` — the same schema
    produced by :meth:`AlpacaLoader.get_ohlcv`.

    Rows where ``close`` is NaN are dropped.  If a ticker fails individually
    it is skipped with a WARNING; if all tickers fail an empty DataFrame with
    the correct MultiIndex and columns is returned.

    This function is intentionally a module-level helper (not a class method)
    so :class:`~equity_signals.data.alpaca_loader.AlpacaLoader` can import it
    without creating a :class:`YFinanceLoader` instance or touching the
    fundamentals cache.

    Args:
        tickers: Uppercase ticker symbols, e.g. ``["AAPL", "MSFT"]``.
        days: Number of calendar days to look back.  Passed as the
            ``period`` argument to ``yf.download`` (e.g. ``"60d"``).

    Returns:
        MultiIndex DataFrame ``(ticker, date)`` with columns
        ``open, high, low, close, volume``.  Empty (but correctly structured)
        if no data could be fetched.
    """
    _OHLCV_COLS = ["open", "high", "low", "close", "volume"]
    _EMPTY = pd.DataFrame(
        columns=_OHLCV_COLS,
        index=pd.MultiIndex.from_tuples([], names=["ticker", "date"]),
    )

    if not tickers:
        return _EMPTY

    parts: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                period=f"{days}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )

            if raw.empty:
                logger.warning("fetch_ohlcv — yfinance returned no data for %s", ticker)
                continue

            # Normalise column names to lowercase.
            raw.columns = [c.lower() for c in raw.columns]

            # Keep only the columns we need; ignore extras (dividends, splits…).
            available = [c for c in _OHLCV_COLS if c in raw.columns]
            raw = raw[available].copy()

            # Drop rows with no close price.
            raw = raw.dropna(subset=["close"])

            if raw.empty:
                logger.warning(
                    "fetch_ohlcv — all rows dropped (NaN close) for %s", ticker
                )
                continue

            # Build MultiIndex (ticker, date).
            raw.index = pd.to_datetime(raw.index).date
            raw.index.name = "date"
            raw.index = pd.MultiIndex.from_arrays(
                [[ticker] * len(raw), raw.index],
                names=["ticker", "date"],
            )

            parts.append(raw)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "fetch_ohlcv — failed to fetch %s (%s: %s); skipping",
                ticker, type(exc).__name__, exc,
            )

    if not parts:
        logger.warning("fetch_ohlcv — no data retrieved for any of: %s", tickers)
        return _EMPTY

    result = pd.concat(parts).sort_index()
    logger.info(
        "yfinance fallback: fetched %d rows for %d ticker(s)",
        len(result),
        result.index.get_level_values("ticker").nunique(),
    )
    return result
