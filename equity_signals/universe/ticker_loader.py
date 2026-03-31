"""ticker_loader — downloads and caches the Russell 2000 constituent list.

:class:`TickerLoader` fetches the iShares IWM ETF holdings CSV, which is the
canonical public source for Russell 2000 constituents, filters it to valid US
equity tickers, and caches the result (ticker + index weight) to disk for one
calendar day.

Two public methods are provided:

* :meth:`~TickerLoader.get_russell2000` — all ~2 000 tickers, sorted
  alphabetically.
* :meth:`~TickerLoader.get_top_pct` — the *top N %* of tickers ranked by
  their index weight, useful for a faster initial scan (default 20 %).

Cache location: ``.cache/russell2000_tickers_YYYYMMDD.csv``
(columns: ``ticker``, ``weight_pct``)

Typical usage::

    from equity_signals.universe.ticker_loader import TickerLoader

    # Full universe (~2 000 tickers)
    tickers = TickerLoader().get_russell2000()

    # Largest 20 % by index weight (~400 tickers)
    tickers = TickerLoader().get_top_pct(20.0)
"""

from __future__ import annotations

import io
import logging
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import ClassVar

import pandas as pd
import requests

from equity_signals.exceptions import TickerLoaderError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IWM_URL: str = (
    "https://www.ishares.com/us/products/239710/"
    "ishares-russell-2000-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IWM_holdings&dataType=fund"
)
_CACHE_DIR: Path = Path(".cache")
_CACHE_PREFIX: str = "russell2000_tickers_"
_CACHE_GLOB: str = "russell2000_tickers_*.csv"
_CACHE_TTL_DAYS: int = 1

# Valid US equity ticker: 1–5 uppercase letters only (no digits, dots, or spaces).
_TICKER_RE: re.Pattern[str] = re.compile(r"^[A-Z]{1,5}$")

# Request headers — iShares occasionally blocks bare user agents.
_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Column name for the index weight in both the IWM CSV and the cache file.
_WEIGHT_COL_CSV: str = "Weight (%)"
_WEIGHT_COL: str = "weight_pct"


# ---------------------------------------------------------------------------
# TickerLoader
# ---------------------------------------------------------------------------


class TickerLoader:
    """Downloads and caches the Russell 2000 equity universe from iShares IWM.

    The iShares IWM ETF holdings CSV is the canonical public source for the
    Russell 2000 index constituents.  The downloaded list is filtered to valid
    US equity tickers, each paired with its index weight, and cached to disk
    for :data:`_CACHE_TTL_DAYS` calendar day(s).

    Args:
        cache_dir: Directory for CSV cache files. Defaults to ``.cache/``.
        url: IWM holdings CSV URL. Override in tests to avoid network calls.

    Example::

        # All tickers
        tickers = TickerLoader().get_russell2000()

        # Top 20 % by index weight (~400 tickers)
        tickers = TickerLoader().get_top_pct(20.0)
    """

    _TICKER_RE: ClassVar[re.Pattern[str]] = _TICKER_RE

    def __init__(
        self,
        *,
        cache_dir: Path = _CACHE_DIR,
        url: str = _IWM_URL,
    ) -> None:
        self._cache_dir = cache_dir
        self._url = url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_russell2000(self) -> list[str]:
        """Return all current Russell 2000 constituent tickers.

        Serves from the on-disk cache when the cached file is less than
        :data:`_CACHE_TTL_DAYS` day(s) old; otherwise downloads, parses,
        and re-caches the IWM holdings CSV.

        Returns:
            Alphabetically-sorted list of valid US equity ticker strings.

        Raises:
            :class:`~equity_signals.exceptions.TickerLoaderError`:
                If the download fails or the CSV cannot be parsed.
        """
        df = self._load_or_fetch()
        tickers = sorted(df["ticker"].tolist())
        logger.info("TickerLoader — %d tickers in Russell 2000 universe", len(tickers))
        return tickers

    def get_top_pct(self, pct: float = 20.0) -> list[str]:
        """Return the top *pct* % of Russell 2000 tickers ranked by index weight.

        Tickers are sorted descending by their ``Weight (%)`` in the IWM ETF.
        The top ``ceil(n × pct / 100)`` are returned, giving a representative
        sample of the largest constituents — typically the stocks with the most
        liquidity and the most reliable fundamental data from yfinance.

        Args:
            pct: Percentage of the universe to keep (0 < pct ≤ 100).
                 Default is ``20.0``, which yields ~400 tickers from the
                 Russell 2000.

        Returns:
            List of ticker strings, ordered by index weight descending (highest
            weight first).

        Raises:
            ValueError: If *pct* is not in the range (0, 100].
            :class:`~equity_signals.exceptions.TickerLoaderError`:
                If the download fails or the CSV cannot be parsed.

        Example::

            top400 = TickerLoader().get_top_pct(20.0)
        """
        if not (0 < pct <= 100):
            raise ValueError(f"pct must be in (0, 100], got {pct}")

        df = self._load_or_fetch()

        # Sort by weight descending; tickers without a weight go to the bottom.
        df_sorted = df.sort_values(_WEIGHT_COL, ascending=False, na_position="last")

        n_total = len(df_sorted)
        n_keep = math.ceil(n_total * pct / 100)
        top_df = df_sorted.head(n_keep)

        tickers = top_df["ticker"].tolist()
        logger.info(
            "TickerLoader — top %.0f%% by index weight: %d / %d tickers selected",
            pct,
            len(tickers),
            n_total,
        )
        return tickers

    # ------------------------------------------------------------------
    # Private — load from cache or fetch from network
    # ------------------------------------------------------------------

    def _load_or_fetch(self) -> pd.DataFrame:
        """Return a DataFrame ``[ticker, weight_pct]``, from cache or network."""
        cached = self._load_cache()
        if cached is not None:
            logger.info(
                "TickerLoader cache hit — %d tickers loaded from disk", len(cached)
            )
            return cached

        logger.info("TickerLoader — downloading IWM holdings from iShares")
        df = self._download_and_parse()
        self._save_cache(df)
        logger.info(
            "TickerLoader — %d valid US equity tickers fetched and cached",
            len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Private — cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, for_date: date) -> Path:
        """Return the CSV cache path for *for_date*."""
        return self._cache_dir / f"{_CACHE_PREFIX}{for_date.strftime('%Y%m%d')}.csv"

    def _load_cache(self) -> pd.DataFrame | None:
        """Return ``DataFrame[ticker, weight_pct]`` from cache, or ``None``."""
        if not self._cache_dir.exists():
            return None

        today = date.today()
        candidates: list[tuple[date, Path]] = []

        for path in self._cache_dir.glob(_CACHE_GLOB):
            date_str = path.stem[len(_CACHE_PREFIX):]
            try:
                file_date = datetime.strptime(date_str, "%Y%m%d").date()
            except ValueError:
                logger.debug("Ignoring unrecognised cache file: %s", path)
                continue
            candidates.append((file_date, path))

        if not candidates:
            return None

        newest_date, newest_path = max(candidates, key=lambda t: t[0])
        age_days = (today - newest_date).days

        if age_days >= _CACHE_TTL_DAYS:
            logger.debug(
                "TickerLoader cache %s is %d day(s) old — re-downloading",
                newest_path.name,
                age_days,
            )
            return None

        try:
            df = pd.read_csv(newest_path)
            # Invalidate old-format caches that lack the weight column.
            if "ticker" not in df.columns or _WEIGHT_COL not in df.columns:
                logger.debug(
                    "Cache %s missing required columns — re-downloading",
                    newest_path.name,
                )
                return None
            df[_WEIGHT_COL] = pd.to_numeric(df[_WEIGHT_COL], errors="coerce")
            logger.debug(
                "Loaded %d tickers from cache %s", len(df), newest_path.name
            )
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read cache file %s: %s", newest_path, exc)
            return None

    def _save_cache(self, df: pd.DataFrame) -> None:
        """Write *df* (columns: ``ticker``, ``weight_pct``) to a dated CSV."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(date.today())
        df.to_csv(path, index=False)
        logger.debug("TickerLoader cache written to %s", path)

    # ------------------------------------------------------------------
    # Private — download and parse
    # ------------------------------------------------------------------

    def _download_and_parse(self) -> pd.DataFrame:
        """Download the IWM CSV and return a DataFrame ``[ticker, weight_pct]``.

        The iShares CSV contains several preamble lines before the actual
        column header row.  This method dynamically locates the header row
        (the first line whose first field is ``"Ticker"``) so the code
        remains robust to changes in the number of preamble rows.

        Returns:
            DataFrame with columns ``ticker`` (str) and ``weight_pct`` (float).
            Rows are sorted by ``weight_pct`` descending.

        Raises:
            :class:`~equity_signals.exceptions.TickerLoaderError`:
                On HTTP errors or if the ``"Ticker"`` column cannot be found.
        """
        try:
            response = requests.get(self._url, headers=_HEADERS, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TickerLoaderError(f"HTTP request failed: {exc}") from exc

        raw_text = response.text
        lines = raw_text.splitlines()

        # Dynamically detect the header row (first row where field 0 == "Ticker").
        header_idx: int | None = None
        for i, line in enumerate(lines):
            first_field = line.split(",")[0].strip().strip('"')
            if first_field == "Ticker":
                header_idx = i
                break

        if header_idx is None:
            raise TickerLoaderError(
                "Could not locate 'Ticker' column header in the IWM holdings CSV. "
                "The iShares CSV format may have changed."
            )

        logger.debug("IWM CSV header found at line %d", header_idx)

        csv_content = "\n".join(lines[header_idx:])
        try:
            raw_df = pd.read_csv(io.StringIO(csv_content))
        except Exception as exc:  # noqa: BLE001
            raise TickerLoaderError(f"Failed to parse IWM CSV: {exc}") from exc

        if "Ticker" not in raw_df.columns:
            raise TickerLoaderError(
                f"'Ticker' column missing after parsing. Available: {list(raw_df.columns)}"
            )

        raw_df["_ticker"] = (
            raw_df["Ticker"].dropna().astype(str).str.strip()
        )
        mask = raw_df["_ticker"].apply(self._is_valid_us_equity)
        filtered = raw_df[mask].copy()

        # Parse index weight; rows without a numeric weight get NaN.
        if _WEIGHT_COL_CSV in filtered.columns:
            filtered[_WEIGHT_COL] = pd.to_numeric(
                filtered[_WEIGHT_COL_CSV], errors="coerce"
            )
        else:
            logger.warning(
                "IWM CSV does not contain '%s' column — weights will be NaN",
                _WEIGHT_COL_CSV,
            )
            filtered[_WEIGHT_COL] = float("nan")

        result = (
            filtered[["_ticker", _WEIGHT_COL]]
            .rename(columns={"_ticker": "ticker"})
            .sort_values(_WEIGHT_COL, ascending=False, na_position="last")
            .reset_index(drop=True)
        )

        logger.debug(
            "IWM CSV — %d raw rows, %d valid US equity tickers after filtering",
            len(raw_df),
            len(result),
        )
        return result

    @staticmethod
    def _is_valid_us_equity(ticker: str) -> bool:
        """Return ``True`` if *ticker* looks like a valid US equity symbol.

        Accepted: 1–5 uppercase ASCII letters only (e.g. ``"AAPL"``, ``"BRK"``).
        Rejected: cash positions (``"-"``), futures, ADR suffixes (``"."``),
        digits, whitespace, and anything longer than 5 characters.
        """
        return bool(_TICKER_RE.fullmatch(ticker))
