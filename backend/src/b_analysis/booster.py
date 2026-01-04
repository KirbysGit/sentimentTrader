import re
from collections import defaultdict

from src.utils.ticker_context import ticker_context, negative_context
from src.utils.config import common_finance_words, popular_tickers, us_states, ambiguous


class Booster:
    def find_mentions(self, combined_raw: str, bare: str, aliases, window_size: int = 200):
        # -- 1. initialize contexts and count.
        contexts = []
        count = 0
        targets = {bare.lower()} | {a.lower() for a in aliases}

        # -- 2. iterate through targets.
        for target in targets:
            # -- 2.1. create pattern.
            pattern = rf"\b{re.escape(target)}\b"
            # -- 2.2. iterate through matches.
            for m in re.finditer(pattern, combined_raw, flags=re.IGNORECASE):
                # -- 2.2.1. increment count.
                count += 1
                # -- 2.2.2. get span and excerpt.
                start, end = m.span()
                excerpt = combined_raw[max(0, start - window_size): min(len(combined_raw), end + window_size)]
                contexts.append(excerpt)

        # de-duplicate excerpts while preserving order for cleaner debug logs
        seen = {}
        for ctx in contexts:
            seen.setdefault(ctx, True)
        unique_contexts = list(seen.keys())

        return count, unique_contexts

    def boost_tickers(self, tickers: set, title: str, text: str) -> tuple[dict, dict]:
        if not tickers:
            return {}, {}

        # -- 1. clean up text.
        combined = f"{title}\n{text}".strip()
        normalized = self.normalize_text(combined)
        title_norm = self.normalize_text(title)
        words = normalized.split()
        debug_hits = []

        mentions = {}

        # -- 2. initialize context.
        finance_ctx = common_finance_words
        popular_ctx = popular_tickers

        scores = {}

        # -- 3. iterate through tickers.
        for t in tickers:

            # -- 3.1. create ticker vars and clean formatting. (ex. tsla)
            t_upper = t.upper()                       # $tsla - > $TSLA
            bare = t_upper.lstrip("$")                # $TSLA - > TSLA
            bare_norm = bare.lower()                  # tsla - > tsla
            has_dollar = t_upper.startswith("$")      # $TSLA - > True

            base = 0

            # -- 3.2. check if ticker is a unit or warrant. (then we just skip.)
            if self.is_unit_or_warrant(bare):
                continue

            # -- 3.3. check if ticker gen formatting.
            #  - if it starts with a $ -> + 3 points.
            #  - if it's a single letter -> + 1 point.
            #  - otherwise -> + 2 points.
            if has_dollar:
                base += 4
            elif len(bare) == 1:
                base += 1
            else:
                base += 2

            # -- 3.4. check if ticker is in title.
            #  - if so -> + 1 point.
            if bare_norm in title_norm:
                base += 1

            has_title_hit = bare_norm in title_norm

            # -- 3.5. check if ticker has finance context.
            #  - if it does -> + 2 points.
            finance_hits = 0
            for i, w in enumerate(words):
                if w == bare_norm:
                    window = words[max(0, i-6): i+7]
                    finance_hits += sum(1 for x in window if x in finance_ctx)

            base += min(2, finance_hits)

            # -- 3.6. get counts of ticker / alias mentions. w/ contexts of mentions.
            #  - if so -> + 2 point.
            aliases = [
                alias for alias, primary in self.aliases.items() if primary == bare
            ]

            mentions, contexts = self.find_mentions(combined, bare_norm, aliases)

            if mentions > 1:
                base += min(2, mentions - 1)

            # -- 3.7. drop address-like state abbreviations (e.g., ", WY")
            #  - if so -> continue.
            if bare in us_states and contexts:
                ctx_has_address = False
                for ctx in contexts:
                    ctx_lower = ctx.lower()
                    if re.search(r",\s*" + re.escape(bare_norm) + r"\b", ctx_lower):
                        ctx_has_address = True
                        break
                if ctx_has_address:
                    continue

            # -- 3.8. negative context: use captured excerpts to penalize if any neg keywords appear
            #  - if so -> - 2 points.
            neg_terms = [kw.lower() for kw in negative_context.get(t_upper, [])]
            if neg_terms and contexts:
                neg_hits = 0
                for ctx in contexts:
                    ctx_lower = ctx.lower()
                    neg_hits += sum(1 for kw in neg_terms if kw in ctx_lower)
                if neg_hits:
                    base = 0

            # -- 3.9. check if ticker has context.
            #  - if it does and has hits -> + 2 points.
            #  - if it does and has no hits -> set score to 0.

            ctx_list = [kw.lower() for kw in ticker_context.get(t_upper, [])]

            ticker_ctx_hits = sum(1 for kw in ctx_list if kw and kw in normalized)

            if bare in ambiguous and ticker_ctx_hits == 0:
                continue

            base += min(5, ticker_ctx_hits)

            # -- 3.8. context/short guards (skip only for ambiguous/short, unless strong signals)

            if len(bare) == 1 and ctx_list and ticker_ctx_hits == 0:
                continue

            if len(bare) == 1 and not ctx_list and bare not in popular_ctx and not has_dollar:
                continue

            if contexts:
                mentions.setdefault(bare, []).extend(contexts)

            bare = t_upper.lstrip("$").strip().upper()
            bare_norm = bare.lower()

            # -- 3.9. add base score to scores dict.
            if base > 0:
                scores[bare] = base

        return scores, mentions

    def clean_boosted(self, boosted: dict, abs_floor: int = 2, rel_pct: float = 0.7) -> dict:
        if not boosted:
            return {}
        max_score = max(boosted.values())
        keep = {}
        for t, s in boosted.items():
            if s >= abs_floor or s >= rel_pct * max_score:
                keep[t] = s
        return keep

