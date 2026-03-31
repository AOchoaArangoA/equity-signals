"""universe_store — helpers for persisting and loading universe snapshots.

:func:`load_latest_universe` finds the most recent ``universe_*.parquet``
file and returns it as a DataFrame.  It searches ``/tmp/`` first (written by
the API's background task), then falls back to ``output/`` (written by the
CLI scripts).  Pass *output_dir* explicitly to override both defaults.

Typical usage::

    from equity_signals.universe.universe_store import load_latest_universe

    universe = load_latest_universe()
"""

from __future__ import annotations

import logging
from glob import glob
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Directories searched in priority order when output_dir is not specified.
_SEARCH_DIRS: list[str] = ["/tmp", "output"]


def load_latest_universe(output_dir: str | None = None) -> pd.DataFrame:
    """Load the most recent ``universe_*.parquet``.

    Search order (when *output_dir* is ``None``):

    1. ``/tmp/universe_*.parquet``  — written by the API background task.
    2. ``output/universe_*.parquet``— written by ``equity-universe-scan`` CLI.

    Files within each directory are sorted lexicographically; because they are
    date-stamped (``universe_YYYYMMDD.parquet``) the last entry is always the
    newest.  The first directory that contains at least one matching file wins.

    Args:
        output_dir: Override the search path.  When given, only that directory
            is searched (no fallback).  Defaults to ``None`` (auto-search).

    Returns:
        DataFrame with at least columns ``ticker``, ``value_signal``, and
        ``pb_rank_sector`` as written by
        :func:`equity_signals.scripts.run_universe_scan.run`.

    Raises:
        FileNotFoundError: If no matching file exists in any search directory.
    """
    search_dirs = [output_dir] if output_dir is not None else _SEARCH_DIRS

    for directory in search_dirs:
        pattern = str(Path(directory) / "universe_*.parquet")
        files = sorted(glob(pattern))
        if files:
            path = files[-1]
            logger.info("Loading universe from %s", path)
            return pd.read_parquet(path)

    searched = ", ".join(f"'{d}'" for d in search_dirs)
    raise FileNotFoundError(
        f"No universe file found in {searched}. "
        "Run 'equity-universe-scan' or POST /api/v1/universe/scan first."
    )
