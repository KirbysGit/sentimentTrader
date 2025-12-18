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
from src.utils.config import suffixes, ticker_stop_terms, common_finance_words, popular_tickers
from src.utils.path_config import tickers_dir, reasoning_dir
from src.utils.ticker_aliases import get_canonical_alias_map
from src.utils.ticker_context import ticker_context

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

    # --- helper methods.

    @staticmethod
    def has_clean_boundary(text, start, end):
        # ensure match is not in middle of a word.
        def is_valid_prev(char):
            if not char:
                return True
            return not char.isalnum() and char != "_"
        def is_valid_next(char):
            if not char:
                return True
            return not char.isalnum() and char != "_"

        prevChar = text[start - 1] if start > 0 else ""
        nextChar = text[end] if end < len(text) else ""

        return is_valid_prev(prevChar) and is_valid_next(nextChar)
    
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

    def find_mentions(self, combined_raw: str, normalized: str, bare: str, aliases):
        contexts = []
        count = 0
        targets = {bare.lower()} | {a.lower() for a in aliases}

        for target in targets:
            pattern = rf"\b{re.escape(target)}\b"
            for m in re.finditer(pattern, combined_raw, flags=re.IGNORECASE):
                count += 1
                start, end = m.span()
                excerpt = combined_raw[max(0, start - 40): min(len(combined_raw), end + 40)]
                contexts.append(excerpt)

        return count, contexts

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
                base += 3
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

            mentions, contexts = self.find_mentions(combined, normalized, bare_norm, aliases)

            if contexts:
                debug_mentions.setdefault(bare, []).extend(contexts)

                
            if mentions > 1:
                base += min(2, mentions - 1)

            # -- 3.7. check if ticker has context.
            #  - if it does and has hits -> + 2 points.
            #  - if it does and has no hits -> set score to 0.

            ctx_list = [kw.lower() for kw in ticker_context.get(t_upper, [])]

            ticker_ctx_hits = sum(1 for kw in ctx_list if kw and kw in normalized)

            base += min(2, ticker_ctx_hits)

            # -- 3.8. context/short guards (skip only for ambiguous/short, unless strong signals)

            if len(bare) == 1 and ctx_list and ticker_ctx_hits == 0:
                continue

            if len(bare) == 1 and not ctx_list and bare not in popular_ctx and not has_dollar:
                continue

            # -- 3.9. add base score to scores dict.
            scores[bare] = base

        return scores, debug_mentions


    # --- main extraction methods.
    def company_names(self, text):
        hits = set()
        if not text:
            return hits

        norm_text = self.normalize_text(text)
        for alias, ticker in self.aliases.items():
            alias_norm = alias.lower()
            if alias_norm in ticker_stop_terms:
                continue
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, norm_text):
                hits.add(ticker)
        return hits

    def extract_tickers(self, text): 

        RAW_REGEX = r"(?<![A-Za-z0-9\$])[A-Z]{1,5}(?![A-Za-z0-9])"              # set up raw ticker regex. (TSLA)
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
            if cleaned in self.equity_universe["symbol"].values:                # check if it exists in equity.
                universe.add(cleaned)                                           # if so, add to cleaned.

        for ticker in tickers:                                                  # -- 4. etf universe check.
            cleaned = ticker.strip().upper()                                    # clean ticker val.
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

        # -- 4. clears out some basic words i've run into.
        no_stops = self.stop_words(candidates)

        # -- 5. boosting.
        boosted, debug_mentions = self.boost_tickers(no_stops, title, text)

        # collect short excerpts for kept tickers.
        debug_hits = []
        for t, contexts in debug_mentions.items():
            for ctx in contexts:
                debug_hits.append({
                    "post_id": row.get("id"),
                    "ticker": t,
                    "context": ctx,
                })

        self._append_debug(debug_hits)

        print(boosted)

        

    def process(self) -> pd.DataFrame:

        # -- 1. get the raw file from today.
        raw_file = Path(self.input_file)

        # -- 2. read the raw file.
        df = pd.read_csv(raw_file)
        self.raw_count = len(df)
        
        # -- 3. iterate through the posts.
        for _, row in df.iterrows():
            self.process_row(row)
                
