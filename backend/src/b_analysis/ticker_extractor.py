# src/b_analysis/ticker_extractor.py

"""
TickerExtractor
---------------
Single source of truth for:
    - Regex-based ticker extraction
    - Filtering invalid/common/ambiguous words
    - EntityLinker validation
    - Confidence scoring

Used by:
    - reddit_data_processor
    - topic_identifier
"""

import re
from typing import Dict, List, Any
from .entity_linker import EntityLinker


class TickerExtractor:
    """Centralized ticker extraction + validation engine."""

    RAW_TICKER_REGEX = r"\b[A-Z]{1,5}\b"
    DOLLAR_TICKER_REGEX = r"\$[A-Za-z]{1,5}"

    def __init__(self):
        self.entity_linker = EntityLinker()

    # ------------------------------------------------------------------
    def extract_tickers(self, text: str) -> Dict[str, Any]:
        """
        Extract and validate tickers from raw post/comment text.
        Returns a dict:

            {
                "tickers": ["NVDA", "AAPL"],
                "scores": {"NVDA": 0.9, "AAPL": 0.5},
                "debug": {...}
            }
        """

        if not text:
            return {"tickers": [], "scores": {}, "debug": {}}

        tickers = set()

        # ---------- 1. Dollar-style tickers ($AAPL) ----------
        for match in re.findall(self.DOLLAR_TICKER_REGEX, text):
            tickers.add(match.replace("$", "").upper())

        # ---------- 2. Raw uppercase tickers (NVDA, TSLA) ----------
        for match in re.findall(self.RAW_TICKER_REGEX, text):
            tickers.add(match.upper())

        validated = []
        scores = {}

        # ---------- 3. Validate via EntityLinker ----------
        for ticker in sorted(tickers):
            is_valid, confidence = self.entity_linker.validate(text, ticker)
            if is_valid:
                validated.append(ticker)
                scores[ticker] = confidence

        return {
            "tickers": validated,
            "scores": scores,
            "debug": {
                "raw_matches": list(tickers),
                "validated": validated,
            }
        }
