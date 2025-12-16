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
from typing import List, Tuple, Dict, Any


# local imports.
from src.utils.config import (
    MACRO_TERMS,
    WSB_SLANG,
    CONTEXT_REQUIRED_TICKERS,
    VALID_ETFS,
    WELL_KNOWN_TICKERS,
    FINANCE_CONTEXT_WORDS,
    NEGATIVE_CONTEXT_PATTERNS,
    COMMON_WORDS,
    ALWAYS_ALLOW,
)
from src.utils.ticker_context_config import TICKER_CONTEXT
from src.utils.ticker_filters import classify_token
from .entity_linker import EntityLinker


COMMON_WORDS_LOWER = {word.lower() for word in COMMON_WORDS}


class TickerExtractor:
    """centralized ticker extraction + validation engine."""

    # regular expressions for ticker extraction.
    RAW_TICKER_REGEX = r"(?<![A-Za-z0-9\$])[A-Z]{1,5}(?![A-Za-z0-9])"
    DOLLAR_TICKER_REGEX = r"(?<![A-Za-z0-9\$])\$[A-Za-z]{1,5}(?![A-Za-z0-9])"

    # initialize ticker extractor.
    def __init__(self):
        self.entity_linker = EntityLinker()

    # ------------------------------------------------------------------
    def extract_tickers(self, text: str) -> Tuple[List[str], List[float], List[dict], List[Dict[str, Any]]]:
        """
        extract and validate tickers from text.

        returns:
            tickers: ["NVDA", "AAPL"]
            scores:  [0.9, 0.5]    <-- aligned with tickers
        """
        if not text:
            return [], [], [], []

        # extract tickers.
        tickers = set()

        # ---------- 1. dollar-style tickers ($AAPL) ----------

        # checks if dollar ticker regex fits point in our text, then if so verifies, it has a clean boundary
        for match in re.finditer(self.DOLLAR_TICKER_REGEX, text):
            if not self._has_clean_boundary(text, match.start(), match.end()):
                continue
            tickers.add(match.group().replace("$", "").upper())

        # ---------- 2. raw uppercase tickers (NVDA, TSLA) ----------

        # checks if matches our ticker regex.
        for match in re.finditer(self.RAW_TICKER_REGEX, text):
            if not self._has_clean_boundary(text, match.start(), match.end()):
                continue
            tickers.add(match.group().upper())

        # ---------- 3. validate via minimal entity linker ----------
        validated = []
        base_scores = []
        evidence_payload: List[Dict[str, Any]] = []

        filtered, review_items = self._filter_noise_tickers(sorted(tickers), text)

        print(filtered)

        for ticker in filtered:
            is_valid, conf, linker_meta = self.entity_linker.validate(text, ticker)
            if is_valid:
                validated.append(ticker)
                base_scores.append(float(conf))  # ensure numeric
                evidence_payload.append({
                    "ticker": ticker,
                    "linker": linker_meta or {},
                })
            else:
                review_items.append(
                    self._make_review_entry(
                        ticker,
                        reason="entity_linker_reject",
                        text=text,
                        extra={"confidence_hint": float(conf)},
                    )
                )

        if not validated:
            return [], [], review_items, []

        # track pre-boost for evidence
        pre_boost_scores = list(base_scores)
        validated, boosted, boost_details = self.entity_linker.boost_confidences(
            text,
            validated,
            base_scores,
        )

        for idx, evidence in enumerate(evidence_payload):
            evidence["pre_boost_score"] = pre_boost_scores[idx]
            evidence["boosted_score"] = boosted[idx]
            evidence["boosts"] = boost_details[idx] if idx < len(boost_details) else []

        return validated, boosted, review_items, evidence_payload

    # ------------------------------------------------------------------
    def _filter_noise_tickers(self, tickers: List[str], text: str) -> Tuple[List[str], List[dict], Dict[str, Dict[str, Any]]]:
        """advanced filtering: blacklist, alias checks, ETF checks, and local context."""
        clean = []
        review = []
        text_low = text.lower() if text else ""
        words = re.findall(r"[a-z0-9$]+", text_low)


        fin_context_words = {w.lower() for w in FINANCE_CONTEXT_WORDS}

        print(tickers)

        # iterate through each ticker.
        for ticker in tickers:

            # ----- 1. initial classification to get rid of obvious noise.
            classification = classify_token(ticker, text_low)

            # --- negative classifications ---

            if classification == "ignored":
                continue

            if classification == "blocked":
                continue

            if classification == "single_letter":
                continue

            if classification == "timestamp":
                continue

            # --- positive classifications ---

            if classification == "known":
                clean.append(ticker)
                continue

            # --- grab context snippet for evidence.
            context_snippet = self._extract_context_snippet(text_low, ticker.lower(), window=160)

            # --- needs to be reviewed ---

            if classification == "unknown_candidate":
                review.append(self._make_review_entry(ticker, "unknown_symbol", context_snippet))
                continue

            if classification == "negative_context":
                review.append(self._make_review_entry(ticker, "negative_context", context_snippet))
                continue

            # --- potential positive classifications w/ context ---

            if classification == "context_required":
                positions = [i for i, w in enumerate(words) if w == ticker_lower]
                
                review_result = self._context_required_review(ticker, text_low, positions, words)
                if review_result == "missing_context":
                    review.append(self._make_review_entry(ticker, "missing_context", context_snippet))
                    continue
                if review_result == "no_financial_context":
                    review.append(self._make_review_entry(ticker, "no_financial_context", context_snippet))
                    continue
                if review_result == "strong_financial_context":
                    clean.append(ticker)
                    continue

        return clean, review

    def _context_required_review(self, ticker: str, text: str) -> dict:

        # if ticker doesn't have alias context, returning missing_context.
        if not self.entity_linker.has_alias_context(text_low, ticker):
            return "missing_context"

        # check if the ticker has strong financial context.
        strong_context_found = False
        finance_hits = set()

        # iterate through each position in the text.
        for pos in positions:
            window = words[max(0, pos - 8): pos + 9]
            window_hits = {w for w in window if w in fin_context_words}
            if window_hits:
                strong_context_found = True
                finance_hits.update(window_hits)
                break

        # if the ticker does not have strong financial context, add it to the review.
        if not strong_context_found:
            return "no_financial_context"

        return "strong_financial_context"

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
        """extract a context snippet from the text w/ radius of 80 chars around ticker"""

        if not text:
            return ""

        idx = text.find(ticker)

        if idx == -1:
            idx = 0

        start = max(0, idx - window // 2)
        end = min(len(text), idx + window // 2)

        snippet = text[start:end].strip()

        return snippet

    @staticmethod
    def _has_clean_boundary(text: str, start: int, end: int) -> bool:
        # ensure the match is not embedded inside a larger word.
        # checks if char before & after ticker is an alphanumeric value or underscore.

        def _is_valid_prev(char: str) -> bool:
            if not char:
                return True

            return not char.isalnum() and char != "_"

        def _is_valid_next(char: str) -> bool:
            if not char:
                return True

            return not char.isalnum() and char != "_"

        prev_char = text[start - 1] if start > 0 else ""
        next_char = text[end] if end < len(text) else ""

        return _is_valid_prev(prev_char) and _is_valid_next(next_char)
