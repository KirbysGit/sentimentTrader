"""Ticker alias configuration for downstream data collection."""

from typing import Dict, List

TICKER_ALIASES: Dict[str, List[str]] = {
    "AAPL": ["APPL"],
    "BRK.A": ["BRKA"],
    "BRK.B": ["BRK", "BRKB"],
    "GOLD": ["GOLD.TO", "GOLD=F"],
    "LLY": ["LILLY"],
    "LSEG": ["LSEG.L"],
    "LVMH": ["MC.PA", "LVMHF"],
    "NVDA": ["NVIDA", "NVDIA", "NIDA", "NVIDIA"],
    "TSM": ["TSMC"],
}


def get_canonical_alias_map() -> Dict[str, str]:
    """
    Build a mapping of every known alias variation → preferred canonical ticker.
    """
    canonical: Dict[str, str] = {}
    for primary, aliases in TICKER_ALIASES.items():
        primary_up = primary.upper()
        canonical[primary_up] = primary_up
        for alias in aliases:
            alias_up = alias.upper()
            canonical[alias_up] = primary_up
    return canonical


def get_alias_chain(symbol: str) -> List[str]:
    """
    Return the ordered list of symbols we should attempt for a given ticker.
    Starts with the requested symbol, then falls back to canonical + alternates.
    """
    if not symbol:
        return []

    symbol_up = symbol.upper()
    alias_map = get_canonical_alias_map()
    canonical = alias_map.get(symbol_up, symbol_up)

    chain: List[str] = []
    if symbol_up not in chain:
        chain.append(symbol_up)
    if canonical not in chain:
        chain.append(canonical)

    for alias in TICKER_ALIASES.get(canonical, []):
        alias_up = alias.upper()
        if alias_up not in chain:
            chain.append(alias_up)

    return chain
