# src/b_analysis/ticker_extractor.py

import re
from typing import List, Tuple
from src.utils.config import (
    MACRO_TERMS,
    WSB_SLANG,
    WSB_FINANCE_BLACKLIST,
    CONTEXT_REQUIRED_TICKERS,
    VALID_ETFS,
    WELL_KNOWN_TICKERS,
)
from .entity_linker import EntityLinker


class TickerExtractor:
    """Centralized ticker extraction + validation engine."""

    RAW_TICKER_REGEX = r"\b[A-Z]{1,5}\b"
    DOLLAR_TICKER_REGEX = r"\$[A-Za-z]{1,5}"

    def __init__(self):
        self.entity_linker = EntityLinker()

    # ------------------------------------------------------------------
    def extract_tickers(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Extract and validate tickers from text.

        RETURNS:
            tickers: ["NVDA", "AAPL"]
            scores:  [0.9, 0.5]    <-- aligned with tickers
        """

        if not text:
            return [], []

        tickers = set()

        # ---------- 1. Dollar-style tickers ($AAPL) ----------
        for match in re.findall(self.DOLLAR_TICKER_REGEX, text):
            tickers.add(match.replace("$", "").upper())

        # ---------- 2. Raw uppercase tickers (NVDA, TSLA) ----------
        for match in re.findall(self.RAW_TICKER_REGEX, text):
            tickers.add(match.upper())

        validated = []
        scores = []

        # ---------- 3. Validate via Minimal EntityLinker ----------
        filtered = self._filter_noise_tickers(sorted(tickers), text)

        for ticker in filtered:
            is_valid, conf = self.entity_linker.validate(text, ticker)
            if is_valid:
                validated.append(ticker)
                scores.append(float(conf))  # ensure numeric

        return validated, scores

    # ------------------------------------------------------------------
    def _filter_noise_tickers(self, tickers: List[str], text: str) -> List[str]:
        """Advanced filtering: blacklist, alias checks, ETF checks, and local context."""
        clean = []
        text_low = text.lower() if text else ""
        words = text_low.split()

        fin_context_words = {
            "stock", "stocks", "market", "share", "shares", "price", "prices",
            "earnings", "revenue", "eps", "analyst", "valuation", "guidance",
            "options", "calls", "puts", "trading", "trade", "invest", "buy", "sell",
            "portfolio", "ticker", "etf", "company", "fund", "short", "long",
        }

        for ticker in tickers:
            if ticker in MACRO_TERMS or ticker in WSB_SLANG or ticker in WSB_FINANCE_BLACKLIST:
                continue
            if len(ticker) == 1 and ticker not in {"F"}:
                continue
            if ticker == "ET" and re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\s*et\b", text_low):
                continue

            if ticker in WELL_KNOWN_TICKERS or ticker in VALID_ETFS:
                clean.append(ticker)
                continue

            if ticker in CONTEXT_REQUIRED_TICKERS:
                if not self.entity_linker.has_alias_context(text_low, ticker):
                    continue

            positions = [i for i, w in enumerate(words) if w == ticker.lower()]
            if not positions:
                continue

            strong_context_found = False
            for pos in positions:
                window = words[max(0, pos - 5): pos + 6]
                if any(w in fin_context_words for w in window):
                    strong_context_found = True
                    break

            if not strong_context_found:
                continue

            clean.append(ticker)

        return clean
