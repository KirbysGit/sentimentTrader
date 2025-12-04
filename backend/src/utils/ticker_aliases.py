"""Ticker alias configuration for downstream data collection."""

from typing import Dict, List

TICKER_ALIASES = {
    # Commodities / international listings
    "GOLD": ["GOLD", "GOLD.TO", "GOLD=F"],
    "USD": ["DX-Y.NYB", "DXY"],
    "BTC": ["BTC-USD", "XBT-USD"],
    "ETH": ["ETH-USD"],
    # Crypto ETFs
    "IBIT": ["IBIT", "IBIT.O"],
    "BITB": ["BITB", "BITB.O"],
    # ETFs and international listings observed in pipeline runs
    "VEQT": ["VEQT.TO"],
    "VHYG": ["VHYG.L"],
    "LSEG": ["LSEG.L"],
    "LVMH": ["MC.PA", "LVMHF"],
    "FTSE": ["^FTSE"],
    # typo guards
    "APPL": ["AAPL"],
    "NVIDA": ["NVDA"],
    "LILLY": ["LLY"],
    "BRK": ["BRK.B", "BRK.A"],
    "NVDIA": ["NVDA"],
    "NIDA": ["NVDA"],
    "TSMC": ["TSM"],
    "SPX": ["^GSPC"],
}


def get_alias_chain(symbol: str) -> List[str]:
    """
    Return the canonical ticker + any alternate symbols we should try.
    Ensures there are no duplicates and preserves order.
    """
    if not symbol:
        return []

    symbol_up = symbol.upper()
    alias_chain = [symbol_up]
    for alias in TICKER_ALIASES.get(symbol_up, []):
        alias_up = alias.upper()
        if alias_up not in alias_chain:
            alias_chain.append(alias_up)
    return alias_chain


def get_canonical_alias_map() -> Dict[str, str]:
    """
    Build a mapping of every known alias variation → preferred canonical ticker.
    """
    canonical: Dict[str, str] = {}
    for primary, aliases in TICKER_ALIASES.items():
        primary_up = primary.upper()
        canonical.setdefault(primary_up, primary_up)
        for alias in aliases:
            alias_up = alias.upper()
            canonical.setdefault(alias_up, primary_up)
    return canonical
