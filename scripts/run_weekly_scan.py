#!/usr/bin/env python3
"""run_weekly_scan — DEPRECATED combined pipeline (Steps 1–4).

.. deprecated::
    Use :mod:`equity_signals.scripts.run_universe_scan` and
    :mod:`equity_signals.scripts.run_signal_scan` instead.
    This script runs both sequentially and is kept for backwards compatibility.

Usage::

    python scripts/run_weekly_scan.py            # still works
    python scripts/run_weekly_scan.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

warnings.warn(
    "run_weekly_scan.py runs both scans sequentially. "
    "Prefer run_universe_scan.py + run_signal_scan.py for production.",
    DeprecationWarning,
    stacklevel=2,
)

from equity_signals.scripts.run_signal_scan import run as _run_signal  # noqa: E402
from equity_signals.scripts.run_universe_scan import run as _run_universe  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "[DEPRECATED] Combined universe + signal scan. "
            "Prefer equity-universe-scan + equity-signal-scan for production."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index-top-pct", type=float, default=20.0, metavar="PCT")
    parser.add_argument("--midcap-min", type=float, default=300_000_000, metavar="USD")
    parser.add_argument("--midcap-max", type=float, default=2_000_000_000, metavar="USD")
    parser.add_argument("--sectors", nargs="*", default=[], metavar="SECTOR")
    parser.add_argument("--pb-percentile", type=float, default=30.0, metavar="PCT")
    parser.add_argument("--top-n", type=int, default=5, metavar="N")
    parser.add_argument("--window", type=int, default=20, metavar="DAYS")
    parser.add_argument("--z-entry", type=float, default=1.5, metavar="Z")
    parser.add_argument("--z-exit", type=float, default=0.5, metavar="Z")
    parser.add_argument("--days", type=int, default=60, metavar="DAYS")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logger.warning(
        "run_weekly_scan.py is deprecated — "
        "use equity-universe-scan + equity-signal-scan instead"
    )

    _run_universe(
        index_top_pct=args.index_top_pct,
        midcap_min=args.midcap_min,
        midcap_max=args.midcap_max,
        sectors=args.sectors or [],
        pb_percentile=args.pb_percentile,
    )

    _run_signal(
        top_n=args.top_n,
        window=args.window,
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        days=args.days,
    )


if __name__ == "__main__":
    main()
