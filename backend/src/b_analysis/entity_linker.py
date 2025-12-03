# src/b_analysis/entity_linker.py

"""
purpose:
  validate extracted tickers, boost or reject them using light context rules,
  and hand a clean symbol list to downstream scoring.

what this does for us:
  - filters obvious junk (common words, macro terms, blocked symbols)
  - auto-accepts ETFs/whitelisted aliases for speed
  - boosts confidence when company keywords or aliases appear in text
  - flags context-required symbols (AI, GDP, etc.) if supporting words are missing

where it fits:
  stage 2 (analysis). TickerExtractor emits raw candidates → EntityLinker
  trims/boosts them → RedditDataProcessor continues with sentiment + diagnostics.

organization:
  - imports
  - EntityLinker class
    - __init__: load config-driven blacklists/whitelists
    - validate: core yes/no + base confidence
    - boost_confidences: apply post-validation score nudges
    - has_alias_context: helper for context keyword checks
"""

# imports.
import logging
from typing import Tuple, List

# local imports.
from src.utils.config import (
    MACRO_TERMS,
    WSB_SLANG,
    CONTEXT_REQUIRED_TICKERS,
    STOCK_DATA_BLACKLIST,
    COMMON_WORDS,
    VALID_ETFS,
)
from src.utils.ticker_aliases import TICKER_ALIASES
from src.utils.ticker_context_config import TICKER_CONTEXT, CONTEXT_BLACKLIST

# setup logging.
logger = logging.getLogger(__name__)


# entity linker class
class EntityLinker:
    """Lightweight context-based validator for extracted ticker symbols."""

    def __init__(self):
        """initialize entity linker with common word blacklist and ETF whitelist."""
        self.common_word_blacklist = set(COMMON_WORDS)
        self.etf_whitelist = set(VALID_ETFS)

        self.ticker_context = TICKER_CONTEXT
        self.context_blacklist = CONTEXT_BLACKLIST

    # ==================================================================
    # core validation
    # ==================================================================
    def validate(self, text: str, ticker: str) -> Tuple[bool, float]:
        """
        decide whether the extracted ticker is real + return base confidence.
        """

        if not text or not ticker:
            return False, 0.0

        text_low = text.lower()
        ticker = ticker.upper()

        # 1. etf = always valid
        if ticker in self.etf_whitelist:
            return True, 1.0

        # 2. blacklist = invalid
        if ticker in self.common_word_blacklist:
            return False, 0.0
        if ticker in MACRO_TERMS or ticker in WSB_SLANG:
            return False, 0.0
        if ticker in STOCK_DATA_BLACKLIST:
            return False, 0.0

        # 3. context blacklist overrides everything.
        blacklist_terms = self.context_blacklist.get(ticker, [])
        if blacklist_terms and any(term in text_low for term in blacklist_terms):
            return False, 0.0

        # 4. context keywords boost confidence.
        if self.has_alias_context(text_low, ticker):
            return True, 0.9

        # 5. configured aliases imply known ticker.
        if ticker in TICKER_ALIASES:
            return True, 0.8

        # 3. context-required tickers (AI, GDP, etc.).
        if ticker in CONTEXT_REQUIRED_TICKERS:
            if not self.has_alias_context(text_low, ticker):
                return False, 0.0

        # 4. literal ticker mention ($TSLA or TSLA in text).
        if ticker.lower() in text_low:
            return True, 0.6

        # default low-confidence validation (could still be real).
        return True, 0.5

    # ==================================================================
    # confidence adjustment after extraction
    # ==================================================================
    def boost_confidences(
        self,
        text: str,
        tickers: List[str],
        scores: List[float]
    ) -> Tuple[List[str], List[float]]:
        """
        lightweight booster:
        - Alias match → +0.20
        - literal name/ticker in text → +0.10
        - very low score → small penalty
        """

        if not tickers:
            return tickers, scores

        text_low = text.lower()
        boosted = []

        for tkr, score in zip(tickers, scores):
            tkr_up = tkr.upper()
            new_score = score

            # 1. boost on literal mention.
            if tkr_up.lower() in text_low:
                new_score = min(new_score + 0.10, 1.0)

            # 2. boost on context keywords.
            for alias in self.ticker_context.get(tkr_up, []):
                if alias in text_low:
                    new_score = min(new_score + 0.20, 1.0)
                    break

            # 3. slight penalty for extremely low confidence.
            if new_score < 0.15:
                new_score = max(new_score - 0.05, 0)

            boosted.append(new_score)

        return tickers, boosted

    # ==================================================================
    # helpers
    # ==================================================================
    def has_alias_context(self, text_low: str, ticker: str) -> bool:
        """
        require supporting company keywords for ambiguous tickers
        like AI, GDP, VAT, etc.
        """
        for alias in self.ticker_context.get(ticker, []):
            if alias in text_low:
                return True
        return False
