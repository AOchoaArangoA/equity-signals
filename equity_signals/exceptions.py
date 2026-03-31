"""exceptions — custom exception hierarchy for the equity-signals package.

All public exceptions raised by this package are defined here so callers can
catch them at the appropriate level of granularity without importing internal
modules.
"""


class EquitySignalsError(Exception):
    """Base class for all equity-signals exceptions."""


# ---------------------------------------------------------------------------
# FMP-specific exceptions
# ---------------------------------------------------------------------------


class FMPError(EquitySignalsError):
    """Base class for errors originating from the FMP loader."""


class FMPRateLimitError(FMPError):
    """Raised when the FMP API returns HTTP 429 and all retries are exhausted.

    Args:
        ticker: The ticker symbol that triggered the rate-limit response.
        attempts: Number of attempts made before giving up.
    """

    def __init__(self, ticker: str, attempts: int) -> None:
        self.ticker = ticker
        self.attempts = attempts
        super().__init__(
            f"FMP rate limit exceeded for '{ticker}' after {attempts} attempt(s)."
        )


class FMPResponseError(FMPError):
    """Raised when the FMP API returns an unexpected HTTP status code.

    Args:
        ticker: The ticker symbol that triggered the error.
        status_code: HTTP status code returned by the API.
        message: Optional detail from the response body.
    """

    def __init__(self, ticker: str, status_code: int, message: str = "") -> None:
        self.ticker = ticker
        self.status_code = status_code
        self.message = message
        detail = f" — {message}" if message else ""
        super().__init__(
            f"FMP API error for '{ticker}': HTTP {status_code}{detail}"
        )


class FMPDataError(FMPError):
    """Raised when the FMP response is successful but the payload is unusable.

    Args:
        ticker: The ticker symbol whose data could not be parsed.
        reason: Human-readable explanation of what was wrong with the payload.
    """

    def __init__(self, ticker: str, reason: str) -> None:
        self.ticker = ticker
        self.reason = reason
        super().__init__(f"FMP data error for '{ticker}': {reason}")


# ---------------------------------------------------------------------------
# TickerLoader-specific exceptions
# ---------------------------------------------------------------------------


class TickerLoaderError(EquitySignalsError):
    """Raised when TickerLoader cannot fetch or parse the Russell 2000 holdings.

    Args:
        reason: Human-readable explanation of what went wrong.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"TickerLoader error: {reason}")


# ---------------------------------------------------------------------------
# Alpaca-specific exceptions
# ---------------------------------------------------------------------------


class AlpacaError(EquitySignalsError):
    """Base class for errors originating from the Alpaca loader."""


class AlpacaAPIError(AlpacaError):
    """Raised when the Alpaca API returns an error response.

    Wraps ``alpaca.common.exceptions.APIError`` with context about which
    tickers were being requested.

    Args:
        tickers: Ticker symbols that were being requested.
        status_code: HTTP status code returned by the API.
        message: Detail message from the API response.
    """

    def __init__(self, tickers: list[str], status_code: int, message: str = "") -> None:
        self.tickers = tickers
        self.status_code = status_code
        self.message = message
        symbols = ", ".join(tickers)
        detail = f" — {message}" if message else ""
        super().__init__(
            f"Alpaca API error for [{symbols}]: HTTP {status_code}{detail}"
        )


class AlpacaDataError(AlpacaError):
    """Raised when the Alpaca response is successful but the payload is unusable.

    Args:
        tickers: Ticker symbols whose data could not be parsed.
        reason: Human-readable explanation of what was wrong with the payload.
    """

    def __init__(self, tickers: list[str], reason: str) -> None:
        self.tickers = tickers
        self.reason = reason
        symbols = ", ".join(tickers)
        super().__init__(f"Alpaca data error for [{symbols}]: {reason}")
