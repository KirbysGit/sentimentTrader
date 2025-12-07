"""Helpers for loading and accessing the exchange-wide ticker universe."""

from __future__ import annotations

from pathlib import Path
from typing import Set

from src.utils.path_config import PROJECT_ROOT

DEFAULT_UNIVERSE_PATH = PROJECT_ROOT / "data" / "ticker_universe.txt"


def load_ticker_universe(path: Path = DEFAULT_UNIVERSE_PATH) -> Set[str]:
    """Load the known ticker universe from a newline-delimited text file."""
    if not path.exists():
        raise FileNotFoundError(f"Ticker universe file not found: {path}")

    symbols: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            symbols.add(line.upper())

    if not symbols:
        raise ValueError(f"Ticker universe file {path} produced an empty set")

    return symbols


TICKER_UNIVERSE: Set[str] = load_ticker_universe()


