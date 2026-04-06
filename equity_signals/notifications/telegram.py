"""equity_signals.notifications.telegram — Telegram bot notification helper.

Sends HTML-formatted messages to a configured Telegram chat via the Bot API.
All failures are caught and logged; this module never raises to the caller.

If ``TELEGRAM_BOT_TOKEN`` or ``TELEGRAM_CHAT_ID`` are not set in the
environment, all send calls are silently skipped.

Typical usage::

    from equity_signals.notifications.telegram import TelegramNotifier

    notifier = TelegramNotifier()
    notifier.send("<b>Hello</b> from equity-signals")
    notifier.send_table("Signals", [
        {"Ticker": "AAPL", "Z": "+1.26", "Signal": "—"},
        {"Ticker": "MSFT", "Z": "-0.65", "Signal": "—"},
    ])
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT  = 10   # seconds per attempt
_RETRIES  = 1    # one retry on failure


class TelegramNotifier:
    """Send Telegram messages via the Bot API.

    Reads ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` from the pydantic-
    settings singleton.  If either is absent the instance is a no-op and logs
    a single WARNING at construction time.
    """

    def __init__(self) -> None:
        from equity_signals.config import settings

        self._token:   str | None = settings.telegram_bot_token
        self._chat_id: str | None = settings.telegram_chat_id
        self._enabled: bool = bool(self._token and self._chat_id)

        if not self._enabled:
            log.warning("Telegram not configured — skipping notification")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, message: str) -> bool:
        """Send *message* (HTML) to the configured chat.

        Args:
            message: HTML-formatted message string.

        Returns:
            ``True`` on success, ``False`` on any failure.
        """
        if not self._enabled:
            return False

        url     = _API_BASE.format(token=self._token)
        payload: dict[str, Any] = {
            "chat_id":    self._chat_id,
            "text":       message,
            "parse_mode": "HTML",
        }

        for attempt in range(_RETRIES + 1):
            try:
                resp = requests.post(url, json=payload, timeout=_TIMEOUT)
                if resp.ok:
                    log.debug("Telegram message sent (attempt %d)", attempt + 1)
                    return True
                log.warning(
                    "Telegram API error (attempt %d): %s %s",
                    attempt + 1, resp.status_code, resp.text[:200],
                )
            except requests.RequestException as exc:
                log.warning("Telegram request failed (attempt %d): %s", attempt + 1, exc)

            if attempt < _RETRIES:
                time.sleep(2)

        log.error("Telegram notification failed after %d attempt(s)", _RETRIES + 1)
        return False

    def send_table(self, title: str, rows: list[dict]) -> bool:
        """Format *rows* as a monospace table and send it.

        Args:
            title: Bold heading shown above the table.
            rows:  List of dicts; keys become column headers on the first row.

        Returns:
            ``True`` on success, ``False`` on any failure.
        """
        if not rows:
            return self.send(f"<b>{title}</b>\n\n(no data)")

        headers = list(rows[0].keys())
        col_widths = [
            max(len(str(h)), *(len(str(r.get(h, ""))) for r in rows))
            for h in headers
        ]

        def _row(values: list[str]) -> str:
            return "  ".join(str(v).ljust(w) for v, w in zip(values, col_widths))

        sep   = "  ".join("-" * w for w in col_widths)
        lines = [_row(headers), sep] + [_row([str(r.get(h, "")) for h in headers]) for r in rows]
        table = "\n".join(lines)

        message = f"<b>{title}</b>\n\n<pre>{table}</pre>"
        return self.send(message)

    def send_test(self) -> bool:
        """Send a connectivity test message.

        Returns:
            ``True`` if the message was delivered, ``False`` otherwise.
        """
        return self.send("✅ equity-signals bot connected successfully")
