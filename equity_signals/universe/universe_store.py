"""universe_store — helpers for persisting and loading universe snapshots.

:func:`load_latest_universe` finds the most recent ``universe_*.parquet``
file written by ``run_universe_scan.py`` and returns it as a DataFrame.
It is the canonical way for downstream scripts (e.g. ``run_signal_scan.py``)
to consume the structural analysis output without re-running it.

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

_DEFAULT_OUTPUT_DIR: str = "output"


def load_latest_universe(output_dir: str = _DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    """Load the most recent ``universe_*.parquet`` from *output_dir*.

    Files are sorted lexicographically by name; because they are date-stamped
    (``universe_YYYYMMDD.parquet``) the last entry is always the newest.

    Args:
        output_dir: Directory to search.  Defaults to ``"output"``.

    Returns:
        DataFrame with at least columns ``ticker``, ``value_signal``, and
        ``pb_rank_sector`` as written by
        :func:`equity_signals.scripts.run_universe_scan.main`.

    Raises:
        FileNotFoundError: If no matching file exists in *output_dir*.
    """
    pattern = str(Path(output_dir) / "universe_*.parquet")
    files = sorted(glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No universe file found in '{output_dir}'. "
            "Run run_universe_scan.py first."
        )

    path = files[-1]
    logger.info("Loading universe from %s", path)
    return pd.read_parquet(path)
