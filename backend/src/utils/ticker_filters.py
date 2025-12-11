"""Helpers for determining whether an uppercase token should be treated as a ticker."""

from __future__ import annotations

from typing import Set

from src.utils.config import (
    BLOCKLIST,
    ALWAYS_ALLOW,
    COMMON_WORDS,
)
from src.utils.ticker_universe import TICKER_UNIVERSE


def is_symbol_candidate(token: str, universe: Set[str] | None = None) -> bool:
    """
    Return True if the provided token is allowed to proceed through ticker processing.
    """
    up = _normalize(token)
    if not up:
        return False

    if up in COMMON_WORDS:
        return False

    if up in BLOCKLIST:
        return False

    if up in ALWAYS_ALLOW:
        return True

    universe = universe or TICKER_UNIVERSE
    if _is_repeated_fill(up, universe):
        return False

    return up in universe


def classify_token(token: str, universe: Set[str] | None = None) -> str:
    """
    Classify a token as:
        - "blocked": explicitly blacklisted
        - "known": present in the ticker universe or always-allow set
        - "unknown_candidate": looks ticker-like but absent from our universe
        - "ignored": everything else
    """
    # normalize token to uppercase.
    up = _normalize(token)
    
    # if token is empty, return ignored.
    if not up:
        return "ignored"

    # if token is a common word, return blocked.
    if up in COMMON_WORDS:
        return "blocked"

    # if token is in the blocklist, return blocked.
    if up in BLOCKLIST:
        return "blocked"

    # if token is in the always allow set, return known.
    if up in ALWAYS_ALLOW:
        return "known"

    # if token is in the ticker universe, return known.
    universe = universe or TICKER_UNIVERSE
    if up in universe:
        return "known"

    # if token is a repeated fill, return ignored.
    if _is_repeated_fill(up, universe):
        return "ignored"

    # if token is a valid ticker, return unknown candidate.
    if up.isalpha() and 1 <= len(up) <= 5:
        return "unknown_candidate"

    # if token is not a valid ticker, return ignored.
    return "ignored"


def _normalize(token: str | None) -> str:
    """normalize a token to uppercase."""
    return (token or "").strip().upper()


def _is_repeated_fill(symbol: str, universe: Set[str]) -> bool:
    """
    return True if symbol is a repeated-letter filler like XXXXX.
    """
    if len(symbol) >= 4 and len(set(symbol)) == 1:
        if symbol not in universe and symbol not in ALWAYS_ALLOW:
            return True
    return False


