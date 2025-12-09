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
        for match in re.finditer(self.DOLLAR_TICKER_REGEX, text):
            if not self._has_clean_boundary(text, match.start(), match.end()):
                continue
            tickers.add(match.group().replace("$", "").upper())

        # ---------- 2. raw uppercase tickers (NVDA, TSLA) ----------
        for match in re.finditer(self.RAW_TICKER_REGEX, text):
            if not self._has_clean_boundary(text, match.start(), match.end()):
                continue
            tickers.add(match.group().upper())

        validated = []
        base_scores = []
        evidence_payload: List[Dict[str, Any]] = []

        # ---------- 3. validate via minimal entity linker ----------
        filtered, review_items, extractor_meta = self._filter_noise_tickers(sorted(tickers), text)

        for ticker in filtered:
            is_valid, conf, linker_meta = self.entity_linker.validate(text, ticker)
            if is_valid:
                validated.append(ticker)
                base_scores.append(float(conf))  # ensure numeric
                evidence_payload.append({
                    "ticker": ticker,
                    "extractor": extractor_meta.get(ticker, {}),
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
        evidence: Dict[str, Dict[str, Any]] = {}

        fin_context_words = {w.lower() for w in FINANCE_CONTEXT_WORDS}

        # iterate through each ticker.
        for ticker in tickers:
            if self._is_common_word(ticker, text):
                continue

            classification = classify_token(ticker)
            if classification == "blocked":
                review.append(self._make_review_entry(ticker, "blacklist_filter", text))
                continue
            if classification == "unknown_candidate":
                review.append(self._make_review_entry(ticker, "unknown_symbol", text))
                continue
            if classification == "ignored":
                continue
            ticker_lower = ticker.lower()
            if self._matches_negative_context(ticker, text_low):
                review.append(self._make_review_entry(ticker, "negative_context", text))
                continue
            # check if the ticker is in the macro terms, WSB slang, WSB finance blacklist, or stock data blacklist.
            if (
                ticker in MACRO_TERMS
                or ticker in WSB_SLANG
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

            positions = [i for i, w in enumerate(words) if w == ticker_lower]
            if not positions:
                review.append(self._make_review_entry(ticker, "not_in_text", text))
                continue

            context_tokens = self._collect_context_tokens(words, positions)
            context_snippet = self._extract_context_snippet(text, ticker, window=160)
            entry_meta = {
                "accept_reason": None,
                "context_tokens": context_tokens,
                "context_snippet": context_snippet,
                "finance_terms": [],
                "context_terms": [],
                "peer_hit": False,
                "peer_ticker": None,
            }

            # check if the ticker is in the well known tickers or valid ETFs.
            if ticker in WELL_KNOWN_TICKERS or ticker in ALWAYS_ALLOW:
                entry_meta["accept_reason"] = "well_known_ticker"
                evidence[ticker] = entry_meta
                clean.append(ticker)
                continue

            # check if the ticker is in the ticker context.
            context_terms = TICKER_CONTEXT.get(ticker, [])
            matched_terms = [term for term in context_terms if term in text_low] if context_terms else []
            if matched_terms:
                entry_meta["context_terms"] = matched_terms
                entry_meta["accept_reason"] = "context_keyword"
                evidence[ticker] = entry_meta
                clean.append(ticker)
                continue

            # check if the ticker is in the context required tickers and does not have alias context.
            if ticker in CONTEXT_REQUIRED_TICKERS:
                if not self.entity_linker.has_alias_context(text_low, ticker):
                    review.append(self._make_review_entry(ticker, "missing_context_keyword", text))
                    continue
                entry_meta["context_terms"] = TICKER_CONTEXT.get(ticker, [])

            # check if the ticker has strong financial context.
            strong_context_found = False
            finance_hits = set()
            for pos in positions:
                window = words[max(0, pos - 8): pos + 9]
                window_hits = {w for w in window if w in fin_context_words}
                if window_hits:
                    strong_context_found = True
                    finance_hits.update(window_hits)
                    break

            # if the ticker does not have strong financial context, add it to the review.
            if not strong_context_found:
                peer_hit, peer_symbol = self._has_peer_ticker_context(ticker, tickers, words, positions)
                if peer_hit:
                    entry_meta["peer_hit"] = True
                    entry_meta["peer_ticker"] = peer_symbol
                    entry_meta["accept_reason"] = "peer_ticker_context"
                    evidence[ticker] = entry_meta
                    clean.append(ticker)
                    continue
                review.append(self._make_review_entry(ticker, "no_financial_context", text))
                continue

            entry_meta["finance_terms"] = sorted(finance_hits)
            entry_meta["accept_reason"] = entry_meta.get("accept_reason") or "finance_keywords"
            evidence[ticker] = entry_meta
            # add the ticker to the clean list.
            clean.append(ticker)

        return clean, review, evidence

    def _has_peer_ticker_context(
        self,
        ticker: str,
        all_tickers: List[str],
        words: List[str],
        positions: List[int],
        radius: int = 10,
    ) -> Tuple[bool, str]:
        """Check if nearby well-known tickers provide implicit context."""
        if not positions:
            return False, ""
        for peer in all_tickers:
            if peer == ticker or peer not in WELL_KNOWN_TICKERS:
                continue
            peer_positions = [i for i, w in enumerate(words) if w == peer.lower()]
            if any(abs(pos - peer_pos) <= radius for pos in positions for peer_pos in peer_positions):
                return True, peer
        return False, ""

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

    @staticmethod
    def _collect_context_tokens(words: List[str], positions: List[int], radius: int = 8, max_tokens: int = 40) -> List[str]:
        """Collect a small window of tokens for evidence."""
        tokens: List[str] = []
        for pos in positions:
            start = max(0, pos - radius)
            end = min(len(words), pos + radius + 1)
            tokens.extend(words[start:end])
            if len(tokens) >= max_tokens:
                break
        return tokens[:max_tokens]

    @staticmethod
    def _has_clean_boundary(text: str, start: int, end: int) -> bool:
        """Ensure the match is not embedded inside a larger word."""
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

    @staticmethod
    def _matches_negative_context(ticker: str, text_low: str) -> bool:
        """Check if the token appears inside known non-financial phrases."""
        patterns = NEGATIVE_CONTEXT_PATTERNS.get(ticker, [])
        if not patterns or not text_low:
            return False
        return any(pattern in text_low for pattern in patterns)

    @staticmethod
    def _is_common_word(ticker: str, text: str) -> bool:
        """Check if the token is a common English word emphasized in uppercase."""
        if not ticker:
            return False
        lower = ticker.lower()
        if lower in COMMON_WORDS_LOWER:
            return True
        if not text:
            return False
        text_low = text.lower()
        pattern = re.compile(rf"\b{re.escape(lower)}\b")
        for match in pattern.finditer(text_low):
            original = text[match.start():match.end()]
            if not original.isupper():
                return True
        return False
