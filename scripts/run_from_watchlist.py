#!/usr/bin/env python3
"""run_from_watchlist — drive test_mvp_local.py from config/watchlist.json.

Reads ``config/watchlist.json`` and builds the appropriate
``test_mvp_local.py`` command for the requested mode, then executes it via
:func:`subprocess.run`.

Modes
-----
entry
    Weekly entry scan (Monday 8 am ET).  Runs with ``--execute`` to submit
    real paper-trading orders.

exit
    Intraday exit check (multiple times per day).  Runs with
    ``--check-exits`` using the configured stop-loss and z-exit thresholds.

signals
    Dry run — computes and prints signals for all watchlist tickers without
    submitting any orders.

Usage::

    python scripts/run_from_watchlist.py entry
    python scripts/run_from_watchlist.py exit
    python scripts/run_from_watchlist.py signals

Config file (``config/watchlist.json``) must exist at the repo root and
contain at minimum a ``tickers`` list and a ``strategy`` dict with keys
``window``, ``z_entry``, ``z_exit``, ``stop_loss``, ``days``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_WATCHLIST = _REPO_ROOT / "config" / "watchlist.json"
_SCRIPT = _REPO_ROOT / "test_mvp_local.py"
_PYTHON = sys.executable  # works in .venv locally and in CI equally

_VALID_MODES = ("entry", "exit", "signals")


def load_watchlist() -> dict:
    """Load and return the parsed watchlist config.

    Returns:
        Parsed JSON dict with ``tickers`` and ``strategy`` keys.

    Raises:
        SystemExit: If the config file is not found or is invalid JSON.
    """
    if not _WATCHLIST.exists():
        print(
            f"ERROR: config file not found: {_WATCHLIST}\n"
            "Create config/watchlist.json before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        return json.loads(_WATCHLIST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON in {_WATCHLIST}: {exc}", file=sys.stderr)
        sys.exit(1)


def build_command(mode: str, config: dict) -> str:
    """Return the shell command string for the given *mode*.

    Args:
        mode: One of ``"entry"``, ``"exit"``, ``"signals"``.
        config: Parsed watchlist dict.

    Returns:
        Full shell command as a single string.
    """
    tickers: list[str] = config["tickers"]
    strategy: dict = config["strategy"]

    tickers_str = " ".join(tickers)
    script = str(_SCRIPT)

    if mode == "entry":
        return (
            f"{_PYTHON} {script}"
            f" --tickers {tickers_str}"
            f" --window {strategy['window']}"
            f" --z-entry {strategy['z_entry']}"
            f" --z-exit {strategy['z_exit']}"
            f" --stop-loss {strategy['stop_loss']}"
            f" --days {strategy['days']}"
            f" --execute"
        )

    if mode == "exit":
        return (
            f"{_PYTHON} {script}"
            f" --tickers {tickers_str}"
            f" --check-exits"
            f" --z-exit {strategy['z_exit']}"
            f" --stop-loss {strategy['stop_loss']}"
        )

    # mode == "signals"
    return (
        f"{_PYTHON} {script}"
        f" --tickers {tickers_str}"
        f" --window {strategy['window']}"
        f" --z-entry {strategy['z_entry']}"
        f" --days {strategy['days']}"
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in _VALID_MODES:
        print(
            f"Usage: python scripts/run_from_watchlist.py <mode>\n"
            f"  mode: {' | '.join(_VALID_MODES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    mode = sys.argv[1]
    config = load_watchlist()
    cmd = build_command(mode, config)

    print(f"[run_from_watchlist] mode={mode}")
    print(f"[run_from_watchlist] $ {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
