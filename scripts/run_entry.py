#!/usr/bin/env python3
"""run_entry — apply confluence filter and submit entry orders.

Reads ``output/signals_latest.json`` (produced by ``run_signals.py``) and
``config/watchlist.json``, applies the P/B-rank + Z-score confluence rule,
then submits market-buy (or limit-buy with ``--extended-hours``) orders for
qualifying tickers not already held.

Signals file must exist and be no older than 4 hours — otherwise exits with
an error message asking the operator to run ``run_signals.py`` first.

Usage::

    python scripts/run_entry.py                   # dry run by default
    python scripts/run_entry.py --dry-run         # explicit dry run
    python scripts/run_entry.py --force-entry     # skip confluence check
    python scripts/run_entry.py --extended-hours  # use limit order
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("run_entry")

_REPO_ROOT = Path(__file__).parent.parent
_WATCHLIST = _REPO_ROOT / "config" / "watchlist.json"
_OUTPUT_DIR = _REPO_ROOT / "output"
_SIGNALS_LATEST = _OUTPUT_DIR / "signals_latest.json"
_SIGNALS_MAX_AGE_H = 4


def load_watchlist() -> dict:
    if not _WATCHLIST.exists():
        log.error("config/watchlist.json not found")
        sys.exit(1)
    return json.loads(_WATCHLIST.read_text())


def load_signals() -> dict:
    """Load signals_latest.json; exit 1 if missing or older than 4 hours."""
    if not _SIGNALS_LATEST.exists():
        log.error("Signals outdated. Run run_signals.py first.")
        sys.exit(1)

    data = json.loads(_SIGNALS_LATEST.read_text())
    run_date = datetime.fromisoformat(data["run_date"].replace("Z", "+00:00"))
    age_h = (datetime.now(timezone.utc) - run_date).total_seconds() / 3600
    if age_h > _SIGNALS_MAX_AGE_H:
        log.error(
            "Signals are %.1f hours old (max %d h). Run run_signals.py first.",
            age_h, _SIGNALS_MAX_AGE_H,
        )
        sys.exit(1)

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit entry orders for confluence signals.")
    parser.add_argument("--pb-rank-max",   type=int,   default=3,    help="Max P/B sector rank (default: 3)")
    parser.add_argument("--position-pct",  type=float, default=0.20, help="Cash fraction per position (default: 0.20)")
    parser.add_argument("--z-entry",       type=float, default=None, help="Z-score entry threshold (overrides watchlist)")
    parser.add_argument("--force-entry",   action="store_true",      help="Skip confluence — enter all watchlist tickers")
    parser.add_argument("--extended-hours",action="store_true",      help="Use limit order (extended hours)")
    parser.add_argument("--dry-run",       action="store_true",      help="Log orders without submitting")
    args = parser.parse_args()

    config   = load_watchlist()
    signals  = load_signals()
    strategy = config.get("strategy", {})
    z_entry  = args.z_entry if args.z_entry is not None else strategy.get("z_entry", 1.5)

    tickers: list[str] = config["tickers"]
    sig_map: dict = signals.get("tickers", {})

    result: dict = {
        "run_date":         datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run":          args.dry_run,
        "entries_triggered": [],
        "skipped":          [],
        "errors":           [],
    }

    # ── Open positions (skip already held) ────────────────────────────────────
    try:
        from equity_signals.execution.alpaca_trader import AlpacaTrader
        trader = AlpacaTrader()
        cash = trader.get_available_cash()
        positions = trader.get_open_positions()
        held = {p["ticker"] for p in positions}
        log.info("Cash: $%.2f | Open positions: %s", cash, sorted(held))
    except Exception as exc:
        log.error("Failed to connect to Alpaca: %s", exc)
        result["errors"].append(str(exc))
        _emit(result)
        sys.exit(1)

    # ── Confluence filter ─────────────────────────────────────────────────────
    candidates: list[str] = []
    for i, ticker in enumerate(tickers):
        if ticker in held:
            log.info("SKIP %s — already held", ticker)
            result["skipped"].append({"ticker": ticker, "reason": "already_held"})
            continue

        sig = sig_map.get(ticker, {})
        z   = sig.get("z_score")
        pb_rank = i + 1  # positional rank within watchlist (1-based)

        if args.force_entry:
            candidates.append(ticker)
            continue

        pb_ok = pb_rank <= args.pb_rank_max
        z_ok  = z is not None and z < -z_entry

        if pb_ok and z_ok:
            candidates.append(ticker)
        else:
            reasons = []
            if not pb_ok:
                reasons.append(f"pb_rank={pb_rank} > {args.pb_rank_max}")
            if not z_ok:
                z_str = f"{z:.3f}" if z is not None else "N/A"
                reasons.append(f"z={z_str} >= -{z_entry}")
            reason = " · ".join(reasons)
            log.info("SKIP %s — %s", ticker, reason)
            result["skipped"].append({"ticker": ticker, "reason": reason})

    if not candidates:
        log.info("No confluence entries this run")
        _emit(result)
        return

    log.info("Candidates: %s", candidates)

    # ── Size and submit ───────────────────────────────────────────────────────
    size_per = cash * args.position_pct / len(candidates)

    for ticker in candidates:
        try:
            price = trader.get_current_price(ticker)
            qty   = math.floor(size_per / price)
            if qty <= 0:
                msg = f"qty=0 (price=${price:.2f} > size=${size_per:.2f})"
                log.warning("SKIP %s — %s", ticker, msg)
                result["skipped"].append({"ticker": ticker, "reason": msg})
                continue

            entry: dict = {
                "ticker":           ticker,
                "qty":              qty,
                "price":            round(price, 4),
                "estimated_value":  round(qty * price, 2),
                "order_type":       "limit_extended" if args.extended_hours else "market",
            }

            if args.dry_run:
                entry["status"] = "dry_run"
                log.info("DRY RUN — BUY %s qty=%d ~$%.2f", ticker, qty, qty * price)
            elif args.extended_hours:
                limit_price = round(price * 1.001, 2)
                order = trader.submit_limit_buy(ticker, qty, limit_price=limit_price)
                entry.update({"limit_price": limit_price, "status": "submitted", **order})
                log.info("LIMIT BUY %s qty=%d limit=$%.2f → %s", ticker, qty, limit_price, order)
            else:
                order = trader.submit_market_buy(ticker, qty)
                entry.update({"status": "submitted", **order})
                log.info("MARKET BUY %s qty=%d → %s", ticker, qty, order)

            result["entries_triggered"].append(entry)

        except Exception as exc:
            log.error("Order failed for %s: %s", ticker, exc)
            result["errors"].append({"ticker": ticker, "error": str(exc)})

    # ── Telegram notification ─────────────────────────────────────────────────
    try:
        from equity_signals.notifications.telegram import TelegramNotifier
        notifier = TelegramNotifier()
        run_date_display = result["run_date"].replace("T", " ").replace("Z", " UTC")

        if result["entries_triggered"]:
            for entry in result["entries_triggered"]:
                mode_str = "DRY RUN" if args.dry_run else "EXECUTED"
                msg = (
                    f"<b>🟢 ENTRY — {entry['ticker']}</b>\n"
                    f"Z-score: {sig_map.get(entry['ticker'], {}).get('z_score', 'N/A')}\n"
                    f"Qty: {entry['qty']} shares @ ~${entry['price']:.2f}\n"
                    f"Est. value: ${entry['estimated_value']:.0f}\n"
                    f"Mode: {mode_str}"
                )
                notifier.send(msg)
        else:
            n = len(tickers)
            notifier.send(
                f"<b>⚪ No entries this week</b>\n"
                f"{run_date_display} — all {n} tickers neutral"
            )
    except Exception as exc:
        log.warning("Telegram notification failed: %s", exc)

    _emit(result)


def _emit(result: dict) -> None:
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
