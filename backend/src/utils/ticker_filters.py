"""Helpers for determining whether an uppercase token should be treated as a ticker."""

from __future__ import annotations

from typing import Set

from src.utils.config import (
    BLOCKLIST,
    ALWAYS_ALLOW,
    COMMON_WORDS,
    WELL_KNOWN_TICKERS,
    NEGATIVE_CONTEXT_PATTERNS,
    TIMESTAMP_TICKERS,
    MACRO_TERMS,
    WSB_SLANG,
    CONTEXT_REQUIRED_TICKERS,
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


def classify_token(token: str, text_low: str | None = None, universe: Set[str] | None = None) -> str:
    """
    Classify a token as:
        - "blocked": explicitly blacklisted
        - "known": present in the ticker universe or always-allow set
        - "unknown_candidate": looks ticker-like but absent from our universe
        - "ignored": everything else
    """
    # normalize token to uppercase.
    up = _normalize(token)
    text_low = text_low or ""
    
    # if token is empty, return ignored.
    if not up:
        return "ignored"

    if len(up) == 1 and not up in {"F"}:
        return "single_letter"

    if up in TIMESTAMP_TICKERS:
        return "timestamp"

    # if token is in the blocklist, return blocked.
    if up in BLOCKLIST or up in MACRO_TERMS or up in COMMON_WORDS or up in WSB_SLANG:
        return "blocked"

    if _matches_negative_context(up, text_low):
        return "negative_context"

    if up in CONTEXT_REQUIRED_TICKERS:
        return "context_required"

    universe = universe or TICKER_UNIVERSE
    if up in universe or up in WELL_KNOWN_TICKERS or up in ALWAYS_ALLOW:
        return "known"

    # if token is a valid ticker, return unknown candidate.
    if up.isalpha() and 1 <= len(up) <= 5:
        return "unknown_candidate"

    # if token is not a valid ticker, return ignored.
    return "ignored"


def _normalize(token: str | None) -> str:
    """normalize a token to uppercase."""
    return (token or "").strip().upper()


def _matches_negative_context(ticker: str, text_low: str) -> bool:
    """Check if the token appears inside known non-financial phrases."""
    patterns = NEGATIVE_CONTEXT_PATTERNS.get(ticker, [])
    if not patterns or not text_low:
        return False
    return any(pattern in text_low for pattern in patterns)