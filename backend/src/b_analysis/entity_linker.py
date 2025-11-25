# src/b_analysis/entity_linker.py

"""
Minimal EntityLinker
--------------------
Purpose:
    - Validate ticker mentions
    - Boost confidence based on simple context clues
    - Zero external calls or caching
    - Fast + predictable + stable

This module intentionally avoids:
    - Web scraping
    - Yahoo Finance lookups
    - Officers/products parsing
    - Industry/sector logic

It plays a *supporting* role to TickerExtractor,
not a primary detection role.
"""

import logging
from typing import Dict, Tuple, List
from src.utils.config import (
    MACRO_TERMS,
    WSB_SLANG,
    CONTEXT_REQUIRED_TICKERS
)

logger = logging.getLogger(__name__)


class EntityLinker:
    """Lightweight context-based validator for extracted ticker symbols."""

    def __init__(self):
        # ------------------------------------------------------------
        # Common-word blacklist (false positives)
        # ------------------------------------------------------------
        self.common_word_blacklist = {
            'A', 'AM', 'AN', 'ARE', 'AS', 'AT', 'BE', 'BY', 'CAN', 'DO', 'FOR',
            'GO', 'HAS', 'HAD', 'HE', 'HER', 'HIS', 'HOW', 'I', 'IF', 'IN', 'IS',
            'IT', 'JOB', 'MAN', 'NEW', 'NO', 'NOT', 'NOW', 'OF', 'ON', 'ONE',
            'OR', 'OUT', 'SEE', 'SHE', 'SO', 'SOME', 'THE', 'THEM', 'THEY',
            'THIS', 'TO', 'UP', 'US', 'WAS', 'WE', 'WERE', 'WHAT', 'WHEN', 'WHO',
            'WITH', 'YOU', 'YOUR'
        }

        # ------------------------------------------------------------
        # ETF whitelist — always valid and high confidence
        # ------------------------------------------------------------
        self.etf_whitelist = {
            'SPY', 'QQQ', 'IWM', 'DIA',
            'VTI', 'VOO', 'BND', 'TLT',
            'XLF', 'XLK', 'XLV', 'XLE', 'XLY', 'XLI',
            'XLP', 'XLB', 'XLC', 'XLU', 'XLRE'
        }

        # ------------------------------------------------------------
        # Minimal alias map for top tickers
        # Expanding this improves context matching
        # ------------------------------------------------------------
        self.aliases: Dict[str, List[str]] = {
            'NVDA': ['nvidia', 'geforce', 'rtx', 'cuda', 'gpu', 'jensen'],
            'AAPL': ['apple', 'iphone', 'ipad', 'macbook', 'mac'],
            'MSFT': ['microsoft', 'windows', 'azure', 'xbox', 'nadella'],
            'META': ['meta', 'facebook', 'instagram', 'whatsapp', 'zuckerberg'],
            'GOOGL': ['google', 'alphabet', 'chrome', 'gmail', 'android'],
            'TSLA': ['tesla', 'elon', 'musk', 'model', 'gigafactory'],
            'AI': ['c3.ai', 'c3ai', 'enterprise ai', 'enterprise-ai'],
        }

    # ==================================================================
    # Core validation
    # ==================================================================
    def validate(self, text: str, ticker: str) -> Tuple[bool, float]:
        """
        Decide whether the extracted ticker is real + return base confidence.
        """

        if not text or not ticker:
            return False, 0.0

        text_low = text.lower()
        ticker = ticker.upper()

        # 1. ETF = always valid
        if ticker in self.etf_whitelist:
            return True, 1.0

        # 2. Blacklist = invalid
        if ticker in self.common_word_blacklist:
            return False, 0.0
        if ticker in MACRO_TERMS or ticker in WSB_SLANG:
            return False, 0.0

        # 3. Context-required tickers (AI, GDP, etc.)
        if ticker in CONTEXT_REQUIRED_TICKERS:
            if not self.has_alias_context(text_low, ticker):
                return False, 0.0

        # 4. Alias/company name match
        if ticker in self.aliases:
            for alias in self.aliases[ticker]:
                if alias in text_low:
                    return True, 0.9

        # 5. Literal ticker mention ($TSLA or TSLA in text)
        if ticker.lower() in text_low:
            return True, 0.6

        # Default low-confidence validation (could still be real)
        return True, 0.5

    # ==================================================================
    # Confidence adjustment after extraction
    # ==================================================================
    def boost_confidences(
        self,
        text: str,
        tickers: List[str],
        scores: List[float]
    ) -> Tuple[List[str], List[float]]:
        """
        Lightweight booster:
        - Alias match → +0.20
        - Literal name/ticker in text → +0.10
        - Very low score → small penalty
        """

        if not tickers:
            return tickers, scores

        text_low = text.lower()
        boosted = []

        for tkr, score in zip(tickers, scores):
            tkr_up = tkr.upper()
            new_score = score

            # 1. Boost on literal mention
            if tkr_up.lower() in text_low:
                new_score = min(new_score + 0.10, 1.0)

            # 2. Boost on alias/company name
            for alias in self.aliases.get(tkr_up, []):
                if alias in text_low:
                    new_score = min(new_score + 0.20, 1.0)
                    break

            # 3. Slight penalty for extremely low confidence
            if new_score < 0.15:
                new_score = max(new_score - 0.05, 0)

            boosted.append(new_score)

        return tickers, boosted

    # ==================================================================
    # Helpers
    # ==================================================================
    def has_alias_context(self, text_low: str, ticker: str) -> bool:
        """
        Require supporting company keywords for ambiguous tickers
        like AI, GDP, VAT, etc.
        """
        for alias in self.aliases.get(ticker, []):
            if alias in text_low:
                return True
        return False
