"""Ticker alias configuration for downstream data collection."""

from typing import Dict, List

ticker_aliases: Dict[str, List[str]] = {
    "AAPL": ["APPL"],
    "ADBE": ["ADOBE"],
    "AMD": ["ADVANCED MICRO DEVICES", "ADV MICRO"],
    "AMZN": ["AMAZON"],
    "AVGO": ["BROADCOM"],
    "BAC": ["BOFA", "BANK OF AMERICA"],
    "BABA": ["ALIBABA"],
    "BRK.A": ["BRKA"],
    "BRK.B": ["BRK", "BRKB"],
    "C": ["CITI", "CITIGROUP"],
    "CRM": ["SALESFORCE"],
    "CVX": ["CHEVRON"],
    "DIS": ["DISNEY", "WALT DISNEY"],
    "GOLD": ["GOLD.TO", "GOLD=F"],
    "GOOGL": ["GOOG", "GOOGLE", "ALPHABET"],
    "GS": ["GOLDMAN", "GOLDMAN SACHS"],
    "HD": ["HOME DEPOT"],
    "IBM": ["INTERNATIONAL BUSINESS MACHINES"],
    "INTC": ["INTEL"],
    "JPM": ["JPMORGAN", "JP MORGAN"],
    "LLY": ["LILLY"],
    "LSEG": ["LSEG.L"],
    "LVMH": ["MC.PA", "LVMHF"],
    "MA": ["MASTERCARD"],
    "META": ["FB", "FACEBOOK"],
    "MSFT": ["MICROSOFT"],
    "NFLX": ["NETFLIX"],
    "NVDA": ["NVIDA", "NVDIA", "NIDA", "NVIDIA"],
    "NVO": ["NOVO", "NOVO NORDISK"],
    "OXY": ["OCCIDENTAL", "OCCIDENTAL PETROLEUM"],
    "ORCL": ["ORACLE"],
    "MU": ["MICRON", "MICRON TECHNOLOGY"],
    "WBD": ["WARNER BROS","WARNER BROTHERS","WARNER BROS DISCOVERY","WARNER BROTHERS DISCOVERY"],
    "PARA": ["PARAMOUNT", "PARAMOUNT GLOBAL"],
    "F": ["FORD", "FORD MOTOR", "FORD MOTOR COMPANY", "F150"],
    "PYPL": ["PAYPAL"],
    "QCOM": ["QUALCOMM"],
    "SHOP": ["SHOPIFY"],
    "TSLA": ["TESLA"],
    "TSM": ["TSMC","TAIWAN SEMI","TAIWAN SEMICONDUCTOR","TAIWAN SEMICONDUCTOR MANUFACTURING",],
    "TCEHY": ["TENCENT"],
    "UNH": ["UNITEDHEALTH", "UNITED HEALTH"],
    "V": ["VISA"],
    "WFC": ["WELLS FARGO"],
    "XOM": ["EXXON", "EXXONMOBIL", "EXXON MOBIL"],
    "NTDOY": ["NTDOF", "NINTENDO", "MARIO", "NINTENDO CO", "NINTENDO COMPANY", "ZELDA", "POKEMON", "KIRBY"]
}


def get_canonical_alias_map() -> Dict[str, str]:
    canonical: Dict[str, str] = {}
    for primary, aliases in ticker_aliases.items():
        canonical[primary] = primary
        for alias in aliases:
            canonical[alias] = primary
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

    for alias in ticker_aliases.get(canonical, []):
        alias_up = alias.upper()
        if alias_up not in chain:
            chain.append(alias_up)

    return chain
