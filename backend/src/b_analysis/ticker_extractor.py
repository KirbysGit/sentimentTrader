"""
purpose:
  normalize all raw ticker mentions from reddit text, aggressively filter noise,
  and return a clean list of candidate symbols plus diagnostics.

how it fits:
  Stage 2 pipeline flow → RedditDataProcessor cleans posts, then calls
  TickerExtractor to pull symbols → EntityLinker validates/boosts them
  → downstream sentiment + stock lookups stay clean.

what it uses:
  - regex passes for `$TICKER` and bare uppercase words
  - blacklist sets from `config.py` (macro terms, slang, stock-data blacklist)
  - context keywords from `ticker_context_config.py`
  - `EntityLinker` for deep alias/context validation
  - finance keyword sets from config to confirm local context windows

main stages:
  1. collect candidates via regex (dollar + raw)
  2. drop obvious junk (blacklists, single letters, timestamps)
  3. auto-allow whitelisted tickers / ETFs / context rules
  4. run windowed context scan for everything else
  5. return validated list + confidence + review_queue entries

future expansion:
  - hook in NER models or FinBERT spans
  - richer context scoring (sentence-level)
  - dynamic whitelist/blacklist growth via diagnostics feedback loop
"""

# imports.
import re
from typing import List, Tuple


# local imports.
from src.utils.config import (
    MACRO_TERMS,
    WSB_SLANG,
    WSB_FINANCE_BLACKLIST,
    CONTEXT_REQUIRED_TICKERS,
    VALID_ETFS,
    WELL_KNOWN_TICKERS,
    STOCK_DATA_BLACKLIST,
    STRONG_FINANCE_WORDS,
    WEAK_FINANCE_WORDS,
)
from src.utils.ticker_context_config import TICKER_CONTEXT
from .entity_linker import EntityLinker


class TickerExtractor:
    """centralized ticker extraction + validation engine."""

    # regular expressions for ticker extraction.
    RAW_TICKER_REGEX = r"\b[A-Z]{1,5}\b"
    DOLLAR_TICKER_REGEX = r"\$[A-Za-z]{1,5}"

    # initialize ticker extractor.
    def __init__(self):
        self.entity_linker = EntityLinker()

    # ------------------------------------------------------------------
    def extract_tickers(self, text: str) -> Tuple[List[str], List[float], List[dict]]:
        """
        extract and validate tickers from text.

        returns:
            tickers: ["NVDA", "AAPL"]
            scores:  [0.9, 0.5]    <-- aligned with tickers
        """
        if not text:
            return [], [], []

        # extract tickers.
        tickers = set()

        # ---------- 1. dollar-style tickers ($AAPL) ----------
        for match in re.findall(self.DOLLAR_TICKER_REGEX, text):
            tickers.add(match.replace("$", "").upper())

        # ---------- 2. raw uppercase tickers (NVDA, TSLA) ----------
        for match in re.findall(self.RAW_TICKER_REGEX, text):
            tickers.add(match.upper())

        validated = []
        scores = []

        # ---------- 3. validate via minimal entity linker ----------
        filtered, review_items = self._filter_noise_tickers(sorted(tickers), text)

        for ticker in filtered:
            is_valid, conf = self.entity_linker.validate(text, ticker)
            if is_valid:
                validated.append(ticker)
                scores.append(float(conf))  # ensure numeric
            else:
                review_items.append(
                    self._make_review_entry(
                        ticker,
                        reason="entity_linker_reject",
                        text=text,
                        extra={"confidence_hint": float(conf)},
                    )
                )

        return validated, scores, review_items

    # ------------------------------------------------------------------
    def _filter_noise_tickers(self, tickers: List[str], text: str) -> Tuple[List[str], List[dict]]:
        """advanced filtering: blacklist, alias checks, ETF checks, and local context."""
        clean = []
        review = []
        text_low = text.lower() if text else ""
        words = re.findall(r"[a-z0-9$]+", text_low)

        fin_context_words = {
            w.lower() for w in (STRONG_FINANCE_WORDS | WEAK_FINANCE_WORDS)
        }

        # iterate through each ticker.
        for ticker in tickers:
            # check if the ticker is in the macro terms, WSB slang, WSB finance blacklist, or stock data blacklist.
            if (
                ticker in MACRO_TERMS
                or ticker in WSB_SLANG
                or ticker in WSB_FINANCE_BLACKLIST
                or ticker in STOCK_DATA_BLACKLIST
            ):
                review.append(self._make_review_entry(ticker, "blacklist_filter", text))
                continue
            # check if the ticker is a single letter and not "F".
            if len(ticker) == 1 and ticker not in {"F"}:
                review.append(self._make_review_entry(ticker, "single_letter"))
                continue
            # check if the ticker is "ET" and there is a timestamp in the text.
            if ticker == "ET" and re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\s*et\b", text_low):
                review.append(self._make_review_entry(ticker, "timestamp_et", text))
                continue

            # check if the ticker is in the well known tickers or valid ETFs.
            if ticker in WELL_KNOWN_TICKERS or ticker in VALID_ETFS:
                clean.append(ticker)
                continue

            # check if the ticker is in the ticker context.
            context_terms = TICKER_CONTEXT.get(ticker, [])
            if context_terms and any(term in text_low for term in context_terms):
                clean.append(ticker)
                continue

            # check if the ticker is in the context required tickers and does not have alias context.
            if ticker in CONTEXT_REQUIRED_TICKERS:
                if not self.entity_linker.has_alias_context(text_low, ticker):
                    review.append(self._make_review_entry(ticker, "missing_context_keyword", text))
                    continue

            # check if the ticker is in the text.
            positions = [i for i, w in enumerate(words) if w == ticker.lower()]
            if not positions:
                review.append(self._make_review_entry(ticker, "not_in_text", text))
                continue

            # check if the ticker has strong financial context.
            strong_context_found = False
            for pos in positions:
                window = words[max(0, pos - 5): pos + 6]
                if any(w in fin_context_words for w in window):
                    strong_context_found = True
                    break

            # if the ticker does not have strong financial context, add it to the review.
            if not strong_context_found:
                review.append(self._make_review_entry(ticker, "no_financial_context", text))
                continue

            # add the ticker to the clean list.
            clean.append(ticker)

        return clean, review

    # ------------------------------------------------------------------
    def _make_review_entry(self, ticker: str, reason: str, text: str = "", extra: dict = None) -> dict:
        """build a standardized review entry for diagnostics."""
        entry = {
            "ticker": ticker,
            "reason": reason,
            "context_snippet": self._extract_context_snippet(text, ticker),
        }
        if extra:
            entry.update(extra)
        return entry

    def _extract_context_snippet(self, text: str, ticker: str, window: int = 80) -> str:
        """extract a context snippet from the text."""
        if not text:
            return ""
        text_lower = text.lower()
        ticker_lower = ticker.lower()
        idx = text_lower.find(ticker_lower)
        if idx == -1:
            idx = 0
        start = max(0, idx - window // 2)
        end = min(len(text), idx + window // 2)
        snippet = text[start:end].strip()
        return snippet
