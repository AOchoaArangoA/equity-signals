"""app.services.llm_service — Anthropic API integration.

:class:`LLMService` wraps the Anthropic Python SDK to provide LLM-powered
commentary on signal output.  It is intentionally minimal: the service accepts
structured signal data and returns a short natural-language interpretation.

Typical usage::

    from app.services.llm_service import LLMService

    service = LLMService()
    commentary = service.interpret_signals(signals_summary)
"""

from __future__ import annotations

import logging

import anthropic

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 512


class LLMService:
    """Wraps the Anthropic API for signal interpretation.

    The client is created lazily on first use so the service can be
    instantiated at startup without an immediate network call.

    Args:
        api_key: Anthropic API key.  Defaults to ``get_settings().anthropic_api_key``.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key: str = api_key or get_settings().anthropic_api_key
        self._client: anthropic.Anthropic | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpret_signals(self, summary: str) -> str:
        """Return a short LLM-generated commentary for the given signal summary.

        Args:
            summary: Plain-text description of current signal state, e.g.
                ``"AAPL: z=-1.8 signal=1, MSFT: z=0.2 signal=0"``.

        Returns:
            A concise (2–4 sentence) natural-language interpretation of the
            signals.  Returns an empty string if the API call fails, so
            callers never raise.
        """
        client = self._get_client()
        prompt = (
            "You are a quantitative analyst assistant. "
            "Given the following mean-reversion Z-score signals, "
            "provide a concise 2-4 sentence interpretation suitable for "
            "a portfolio manager. Focus on which tickers have actionable "
            "long signals and the overall market context implied by the data.\n\n"
            f"Signal summary:\n{summary}"
        )

        try:
            message = client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text: str = message.content[0].text
            logger.info("LLMService — received %d chars of commentary", len(text))
            return text
        except anthropic.APIError as exc:
            logger.warning("LLMService — Anthropic API error: %s", exc)
            return ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLMService — unexpected error: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_client(self) -> anthropic.Anthropic:
        """Lazily create and cache the Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client
