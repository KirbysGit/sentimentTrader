# 
# imports.
import re
import json
import string
import pandas as pd
from pathlib import Path
from colorama import Fore, Style
from collections import defaultdict

# local imports.
from src.utils.config import suffixes, ticker_stop_terms, common_finance_words, popular_tickers, months, us_states, time_tokens, ambiguous
from src.utils.path_config import tickers_dir, reasoning_dir
from src.utils.ticker_aliases import get_canonical_alias_map
from src.utils.ticker_context import ticker_context, negative_context

class RedditProcessor:

    # --- self-initialize.

    def __init__(self, input_file: Path):

        # -- 1. get input file from phase 1.
        self.input_file = input_file

        # -- 2. set up data from etfs and equities we want to reference.
        self.etf_universe = pd.read_csv(tickers_dir / "etfs.csv")
        self.equity_universe = pd.read_csv(tickers_dir / "equities.csv")

        # -- 3. preprocess universes.
        self.suffixes = suffixes
        self.ticker_by_name, self.aliases = self.build_name_maps(self.equity_universe)

        # -- 4. ticker debugging funnel.
        self.tickers_extraction = 0
        self.ticker_stops = 0
        self.tickers_boosted = 0
        self.posts_with_tickers = 0

        # -- 5. ticker processing info.
        self.agg_scores = defaultdict(int)
        self.agg_counts = defaultdict(int)

    # --- helper methods.

    @staticmethod
    def has_clean_boundary(text, start, end):
        # ensure match is not in middle of a word.
        def is_valid_prev(char):
            if char in {"'", "’"}:
                return False
            if not char:
                return True
            return not char.isalnum() and char != "_"
        def is_valid_next(char):
            if char in {"'", "’"}:
                return False
            if not char:
                return True
            return not char.isalnum() and char != "_"

        prevChar = text[start - 1] if start > 0 else ""
        nextChar = text[end] if end < len(text) else ""

        return is_valid_prev(prevChar) and is_valid_next(nextChar)
    
    @staticmethod
    def _append_debug(records: list):
        # -- 1. check if records is empty.
        if not records:
            return

        # -- 2. get debug path.
        debug_path = reasoning_dir / "reasoning.json"

        # -- 3. create debug path if it doesn't exist.
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        # -- 4. load existing aggregated data (if any).
        aggregated = {}
        if debug_path.exists():
            try:
                with debug_path.open("r", encoding="utf-8") as f:
                    aggregated = json.load(f) or {}
            except Exception:
                aggregated = {}

        # -- 5. group new records by ticker and merge.
        grouped = defaultdict(list)
        for rec in records:
            ticker = rec.get("ticker")
            if not ticker:
                continue
            grouped[ticker].append({k: v for k, v in rec.items() if k != "ticker"})

        for ticker, items in grouped.items():
            aggregated.setdefault(ticker, []).extend(items)

        # -- 6. overwrite file with the merged aggregate.
        with debug_path.open("w", encoding="utf-8") as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)

    def normalize_name(self, s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.lower()
        s = s.translate(str.maketrans("", "", string.punctuation))
        for suf in self.suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
        return " ".join(s.split())        

    def normalize_text(self, s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.lower()
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def build_name_maps(self, df: pd.DataFrame) -> tuple[dict, dict]:
        ticker_by_name = {}
        aliases = {}

        for _, row in df.iterrows():
            sym = str(row.get("symbol", "")).strip().upper()
            name = self.normalize_name(row.get("name", ""))
            if not sym or not name:
                continue
            ticker_by_name[name] = sym
            aliases[name] = sym
        
        canonical_aliases = get_canonical_alias_map()
        for alias, primary in canonical_aliases.items():
            norm_alias = self.normalize_name(alias)
            if norm_alias:
                aliases[norm_alias] = primary

        return ticker_by_name, aliases

    def stop_words(self, tickers: set) -> dict:
        if not tickers:
            return set()

        return {t for t in tickers if t.lower() not in ticker_stop_terms}
        
    def is_unit_or_warrant(self, ticker: str) -> bool:
        UNIT_WARRANT_REGEX = re.compile(r"(?:-W[T]?|-U[N]?|\.W[S]?|\.U|/W[S]?|/U)$")
        return bool(UNIT_WARRANT_REGEX.search(ticker))

    def find_mentions(self, combined_raw: str, bare: str, aliases):
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
                excerpt = combined_raw[max(0, start - 40): min(len(combined_raw), end + 40)]
                contexts.append(excerpt)

        # de-duplicate excerpts while preserving order for cleaner debug logs
        seen = {}
        for ctx in contexts:
            seen.setdefault(ctx, True)
        unique_contexts = list(seen.keys())

        return count, unique_contexts
        
    def is_date_like(self, tok: str, raw: str) -> bool:
        if tok not in months:
            return False
        patt = rf"(?:\b{tok}\b\s*\d{{1,2}}|\b\d{{1,2}}\s*\b{tok}\b)"
        return bool(re.search(patt, raw, flags=re.IGNORECASE))

    def is_time_like(self, tok: str, raw: str) -> bool:
        if tok not in time_tokens:
            return False
        patt = rf"(?:\b\d{{1,2}}:\d{{2}}\s*\b{tok}\b|\b{tok}\b\s*\d{{1,2}}:\d{{2}}|\b\d{{1,2}}\s*\b{tok}\b|\b{tok}\b\s*\d{{1,2}})"
        return bool(re.search(patt, raw, flags=re.IGNORECASE))
        
    def company_names(self, text):
        # -- 1. initialize hits.
        hits = set()
        if not text:
            return hits

        # -- 2. normalize text.
        norm_text = self.normalize_text(text)

        # -- 3. iterate through aliases.
        for alias, ticker in self.aliases.items():

            alias_norm = alias.lower()
            if alias_norm in ticker_stop_terms:
                continue
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, norm_text):
                hits.add(ticker)
        return hits

    def debug_reasonings(self, row, debug):
        if not debug:
            return
        
        debug_hits = []
        for t, contexts in debug.items():
            for ctx in contexts:
                debug_hits.append({
                    "post_id": row.get("id"),
                    "ticker": t,
                    "context": ctx,
                })

        self._append_debug(debug_hits)

    # --- main boosting methods.
    
    def boost_tickers(self, tickers: set, title: str, text: str) -> tuple[dict, dict]:
        if not tickers:                                          
            return {}, {}
        
        # -- 1. clean up text.
        combined = f"{title}\n{text}".strip()               
        normalized = self.normalize_text(combined)
        title_norm = self.normalize_text(title)
        words = normalized.split()
        debug_hits = []

        debug_mentions = {}

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

            if bare in ambiguous:
                if ticker_ctx_hits == 0:
                    continue

            base += min(5, ticker_ctx_hits)

            # -- 3.8. context/short guards (skip only for ambiguous/short, unless strong signals)

            if len(bare) == 1 and ctx_list and ticker_ctx_hits == 0:
                continue

            if len(bare) == 1 and not ctx_list and bare not in popular_ctx and not has_dollar:
                continue

            if contexts:
                debug_mentions.setdefault(bare, []).extend(contexts)

            # -- 3.9. add base score to scores dict.
            if base > 0:
                scores[bare] = base

        return scores, debug_mentions

    def clean_boosted(self, boosted: dict, abs_floor: int = 2, rel_pct: float = 0.7) -> dict:
        if not boosted:
            return {}
        max_score = max(boosted.values())
        keep = {}
        for t, s in boosted.items():
            if s >= abs_floor or s >= rel_pct * max_score:
                keep[t] = s
        return keep

    # --- main extraction methods.

    def extract_tickers(self, text): 

        RAW_REGEX = r"(?<![A-Za-z0-9\$/'])[A-Z]{1,5}(?![A-Za-z0-9])"            # set up raw ticker regex. (TSLA)
        DOLLAR_REGEX = r"(?<![A-Za-z0-9\$])\$[A-Za-z]{1,5}(?![A-Za-z0-9])"      # set up dollar ticker regex. ($TSLA)
        
        tickers = set()

        for match in re.finditer(DOLLAR_REGEX, text):                           # -- 1. dollar-style tickers. ($AAPL)
            if not self.has_clean_boundary(text, match.start(), match.end()):   # if not a clean boundary.
                continue                                                        # skip it.
            tickers.add(match.group().upper())                 # if so, add ticker but get rid of $.

        for match in re.finditer(RAW_REGEX, text):                              # -- 2. raw uppercase tickers. (AAPL, NVDA)
            if not self.has_clean_boundary(text, match.start(), match.end()):   # if not a clean boundary.
                continue                                                        # skip it.
            tickers.add(match.group().upper())                                  # if so, add ticker.

        universe = set()

        for ticker in tickers:                                                  # -- 3. equity universe check.
            cleaned = ticker.strip().upper()                                    # clean ticker val.
            if self.is_date_like(cleaned, text):                                # skip month+day noise.
                continue
            if self.is_time_like(cleaned, text):                                # skip time-like noise (e.g., 4:50 PM)
                continue
            if cleaned in self.equity_universe["symbol"].values:                # check if it exists in equity.
                universe.add(cleaned)                                           # if so, add to cleaned.

        for ticker in tickers:                                                  # -- 4. etf universe check.
            cleaned = ticker.strip().upper()                                    # clean ticker val.
            if self.is_date_like(cleaned, text):                                # skip month+day noise.
                continue
            if self.is_time_like(cleaned, text):                                # skip time-like noise (e.g., 4:50 PM)
                continue
            if cleaned in self.etf_universe["symbol"].values:                   # check if it exists in etf.
                universe.add(cleaned)                                           # if so, add to cleaned.

        return universe

    # --- main processing methods ---

    def process_row(self, row: pd.Series):
        # -- 1. grab the title and text for extraction.
        title_col = row.get("title", "")
        text_col = row.get("text", "")
        title = "" if not isinstance(title_col, str) else title_col
        text = "" if not isinstance(text_col, str) else text_col

        combined = f"{title}\n{text}".strip()
        
        # -- 2. extract tickers.
        ticker_hits = self.extract_tickers(combined)

        # -- 3. name checking.
        name_hits = self.company_names(combined)
        
        candidates = ticker_hits | name_hits

        self.tickers_extraction += len(candidates)

        # -- 4. clears out some basic words i've run into.
        no_stops = self.stop_words(candidates)

        self.ticker_stops += len(no_stops)

        # -- 5. boosting.
        boosted, debug_mentions = self.boost_tickers(no_stops, title, text)

        # -- 6. per-post filtering: keep only meaningful tickers for this post.
        cleaned = self.clean_boosted(boosted, abs_floor=2, rel_pct=0.7)
        if not cleaned:
            return

        # counters.
        self.posts_with_tickers += 1
        self.tickers_boosted += len(boosted)

        # -- 6.5 debug reasoning (only kept tickers).
        kept_mentions = {t: debug_mentions.get(t, []) for t in cleaned}
        self.debug_reasonings(row, kept_mentions)
        for t,s in cleaned.items():
            self.agg_scores[t] += s
            self.agg_counts[t] += 1

        
        

    def process(self) -> pd.DataFrame:

        # raw data looks like :
        # - created_utc, id, subreddit, flair, score, upvote_ratio, num_comments
        #   title, text, link

        print(f"{Fore.CYAN}=== stage 2 : reddit data processing ==={Style.RESET_ALL}")
    
        # -- 1. get the raw file from today.
        raw_file = Path(self.input_file)
    
        # -- 2. read the raw file.
        df = pd.read_csv(raw_file)
        self.raw_count = len(df)
        print(f"just loaded {Fore.YELLOW}{len(df)}{Style.RESET_ALL} raw posts from {Fore.YELLOW}{raw_file.name}{Style.RESET_ALL}")

        # -- 3. iterate through the posts.
        for _, row in df.iterrows():
            self.process_row(row)

        top = sorted(self.agg_scores.items(), key=lambda x: x[1], reverse=True)
        print("tickers by total score (all):")
        for t, s in top:
            print(f"  {t}: {s}")

        top_by_posts = sorted(self.agg_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"top 10 ticker by total posts: {Fore.GREEN}{top_by_posts[:10]}{Style.RESET_ALL}")

        print(f"posts with ≥1 kept tickers: {Fore.GREEN}{self.posts_with_tickers}{Style.RESET_ALL} / {Fore.YELLOW}{self.raw_count}{Style.RESET_ALL}")
        print(f"processed {Fore.GREEN}{self.tickers_extraction}{Style.RESET_ALL} tickers from {Fore.YELLOW}{self.posts_with_tickers}{Style.RESET_ALL} posts with tickers")
        print(f"kept {Fore.GREEN}{self.ticker_stops}{Style.RESET_ALL} tickers after stop-words filter from {Fore.YELLOW}{self.tickers_extraction}{Style.RESET_ALL} candidates")
        print(f"kept {Fore.GREEN}{self.tickers_boosted}{Style.RESET_ALL} tickers after boosting (score > 0) from {Fore.YELLOW}{self.ticker_stops}{Style.RESET_ALL} candidates")
