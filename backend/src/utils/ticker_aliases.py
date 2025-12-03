"""Ticker alias configuration for downstream data collection."""

from typing import List

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

    symbol = symbol.upper()
    alias_chain = [symbol]
    for alias in TICKER_ALIASES.get(symbol, []):
        alias_up = alias.upper()
        if alias_up not in alias_chain:
            alias_chain.append(alias_up)
    return alias_chain

