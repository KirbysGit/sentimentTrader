# imports.
import re
import json
import string
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from colorama import Fore, Style
from collections import defaultdict

# local imports.
from src.utils.config import suffixes, ticker_stop_terms, months, us_states, time_tokens, ambiguous, comment_weight
from src.utils.path_config import tickers_dir, reasoning_dir, processed_reddit_by_day_dir
from src.utils.ticker_aliases import get_canonical_alias_map
from src.utils.ticker_context import ticker_context, negative_context

# helper imports.
from src.b_analysis.helpers.booster import Booster
from src.b_analysis.helpers.sentiment_scorer import SentimentScorer

class RedditProcessor(Booster):

    # --- self-initialize.

    def __init__(self):

        # -- 1. set up data from etfs and equities we want to reference.
        self.etf_universe = pd.read_csv(tickers_dir / "etfs.csv")
        self.equity_universe = pd.read_csv(tickers_dir / "equities.csv")

        # -- 2. preprocess universes.
        self.suffixes = suffixes
        self.ticker_by_name, self.aliases = self.build_name_maps(self.equity_universe)

        # -- 3. ticker debugging funnel.
        self.tickers_extraction = 0
        self.ticker_stops = 0
        self.tickers_boosted = 0
        self.posts_with_tickers = 0
        self.raw_count = 0

        # -- 4. ticker processing info.
        self.agg_scores = defaultdict(int)
        self.agg_counts = defaultdict(int)

        # -- 5. sentiment scorer.
        self.sentiment_scorer = SentimentScorer()

        # -- 6. post-ticker records (for output).
        self.records = []
        self.posts = []

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
        # guard against corrupted / wrong-shaped files (e.g. number, list, etc.)
        if not isinstance(aggregated, dict):
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
        comments_col = row.get("comments_text", "")
        title = "" if not isinstance(title_col, str) else title_col
        text = "" if not isinstance(text_col, str) else text_col
        comments_text = "" if not isinstance(comments_col, str) else comments_col

        combined = f"{title}\n{text}\n{comments_text}".strip()
        
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
        # pass comments alongside body text (keeps title separate for boosting logic).
        full_text = f"{text}\n{comments_text}".strip()
        boosted, debug_mentions = self.boost_tickers(no_stops, title, full_text)

        # -- 6. per-post filtering: keep only meaningful tickers for this post.
        cleaned = self.clean_boosted(boosted, abs_floor=2, rel_pct=0.7)
        if not cleaned:
            return

        # -- 6.25 just add posts to our post lists for later reference.
        # We also keep created_at and basic metrics so we can refresh engagement later (re-poll last N days).
        score = row.get("score", 0) or 0
        num_comments = row.get("num_comments", 0) or 0
        self.posts.append(
            {
                "post_id": row.get("id"),
                "created_at": row.get("created_at"),
                "subreddit": row.get("subreddit", ""),
                "score": score,
                "num_comments": num_comments,
                "upvote_ratio": row.get("upvote_ratio", 0.0),
                "title": row.get("title", ""),
                "text": row.get("text", ""),
                "link": row.get("link", ""),
                "engagement": score + (num_comments * comment_weight),
            }
        )

        # counters.
        self.posts_with_tickers += 1
        self.tickers_boosted += len(boosted)

        # -- 6.5 debug reasoning (only kept tickers).
        kept_mentions = {t: debug_mentions.get(t, []) for t in cleaned}
        self.debug_reasonings(row, kept_mentions)
        for t,s in cleaned.items():
            self.agg_scores[t] += s
            self.agg_counts[t] += 1

        # -- 7. score sentiment per ticker (per context window).
        subreddit = row.get("subreddit", "")
        for ticker, boost_score in cleaned.items():
            contexts = kept_mentions.get(ticker, [])
            if not contexts:
                contexts = [combined]  # fallback: whole post

            for idx, ctx in enumerate(contexts):
                sentiment_result = self.sentiment_scorer.score(ctx, subreddit=subreddit)

                self.records.append({
                    "created_at": row.get("created_at"),
                    "subreddit": subreddit,
                    "score": row.get("score", 0),
                    "num_comments": row.get("num_comments", 0),
                    "upvote_ratio": row.get("upvote_ratio", 0.0),
                    "post_id": row.get("id"),
                    "ticker": ticker,
                    "boost_score": boost_score,
                    "mention_idx": idx,
                    "sentiment_score": sentiment_result["score"],
                    "sentiment_category": sentiment_result["category"],
                    "sentiment_model": sentiment_result["model_used"],
                    "sentiment_context": ctx,
                })

    
    def process(self, df: pd.DataFrame, run_id: str) -> pd.DataFrame:

        # raw data looks like :
        # - created_utc, id, subreddit, flair, score, upvote_ratio, num_comments
        #   title, text, link

        print(f"{Fore.CYAN}=== stage 2 : reddit data processing ==={Style.RESET_ALL}")
    
        # taking a count of the raw data.
        self.raw_count = 0 if df is None else len(df)
    
        # -- 1. iterate through the posts.
        for _, row in df.iterrows():
            self.process_row(row)

        # -- 2. sort the tickers by score.
        top = sorted(self.agg_scores.items(), key=lambda x: x[1], reverse=True)

        top_by_posts = sorted(self.agg_counts.items(), key=lambda x: x[1], reverse=True)

        # bunch of debugging info.
        print(f"top 10 ticker by total posts: {Fore.GREEN}{top_by_posts[:10]}{Style.RESET_ALL}")
        print(f"posts with 1 or more kept tickers: {Fore.GREEN}{self.posts_with_tickers}{Style.RESET_ALL} / {Fore.YELLOW}{self.raw_count}{Style.RESET_ALL}")
        print(f"processed {Fore.GREEN}{self.tickers_extraction}{Style.RESET_ALL} tickers from {Fore.YELLOW}{self.posts_with_tickers}{Style.RESET_ALL} posts with tickers")
        print(f"kept {Fore.GREEN}{self.ticker_stops}{Style.RESET_ALL} tickers after stop-words filter from {Fore.YELLOW}{self.tickers_extraction}{Style.RESET_ALL} candidates")
        print(f"kept {Fore.GREEN}{self.tickers_boosted}{Style.RESET_ALL} tickers after boosting (score > 0) from {Fore.YELLOW}{self.ticker_stops}{Style.RESET_ALL} candidates")

        # --- output 1: posts df (post_id, title, text) ---
        self.posts_df = pd.DataFrame(self.posts).drop_duplicates(subset=["post_id"], keep="last")
        processed_reddit_by_day_dir.mkdir(parents=True, exist_ok=True)
        posts_path = processed_reddit_by_day_dir / f"posts_{run_id}.csv"
        self.posts_df.to_csv(posts_path, index=False)

        # Update master posts table (append + dedupe by post_id, keeping highest engagement snapshot).
        # This powers "refresh last N days" so daily aggregates can be corrected as posts gain traction.
        posts_master_path = processed_reddit_by_day_dir / "posts_all.csv"
        try:
            if posts_master_path.exists():
                old_posts = pd.read_csv(posts_master_path)
                combined_posts = pd.concat([old_posts, self.posts_df], ignore_index=True)
            else:
                combined_posts = self.posts_df.copy()

            # Normalize types + compute engagement defensively.
            combined_posts["post_id"] = combined_posts.get("post_id", "").astype(str)
            combined_posts["created_at"] = combined_posts.get("created_at", "").astype(str)
            combined_posts["score"] = pd.to_numeric(combined_posts.get("score", 0), errors="coerce").fillna(0)
            combined_posts["num_comments"] = pd.to_numeric(combined_posts.get("num_comments", 0), errors="coerce").fillna(0)
            combined_posts["engagement"] = (
                pd.to_numeric(combined_posts.get("engagement", 0), errors="coerce").fillna(0)
            )
            # Prefer recomputing engagement from score/comments if engagement is missing/zero.
            recomputed = combined_posts["score"] + (combined_posts["num_comments"] * float(comment_weight))
            combined_posts["engagement"] = combined_posts["engagement"].where(combined_posts["engagement"] > 0, recomputed)

            # Keep the snapshot with the highest engagement for each post.
            combined_posts = combined_posts.sort_values(["post_id", "engagement"]).drop_duplicates(
                subset=["post_id"], keep="last"
            )
            combined_posts = combined_posts.reset_index(drop=True)
            combined_posts.to_csv(posts_master_path, index=False)
        except Exception:
            # non-fatal
            pass

        # --- output 2: quantitative df (post_id+ticker metrics) ---
        if not self.records:
            return pd.DataFrame()

        mention_df = pd.DataFrame(self.records)
        tmp = mention_df.copy()
            tmp["created_at"] = pd.to_datetime(tmp["created_at"], utc=True, errors="coerce")
            tmp["created_date"] = tmp["created_at"].dt.date.astype(str)
        tmp["score"] = pd.to_numeric(tmp.get("score", 0), errors="coerce").fillna(0)
        tmp["num_comments"] = pd.to_numeric(tmp.get("num_comments", 0), errors="coerce").fillna(0)
        tmp["engagement"] = tmp["score"] + (tmp["num_comments"] * float(comment_weight))

            post_ticker_df = (
                tmp.groupby(["created_date", "post_id", "ticker"], as_index=False)
                .agg(
                    created_at=("created_at", "min"),
                    subreddit=("subreddit", "first"),
                    engagement=("engagement", "max"),
                    boost_score=("boost_score", "max"),
                    post_sentiment=("sentiment_score", "mean"),
                )
            )
            post_ticker_df["created_at"] = post_ticker_df["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

        # --- output 3: daily ticker table + historical master ---
        
        # -- set up the daily dataframe.
            post_ticker_df["weighted_numer"] = post_ticker_df["post_sentiment"] * post_ticker_df["engagement"]
            daily_df = (
                post_ticker_df.groupby(["created_date", "ticker"], as_index=False)
                .agg(
                    mention_count=("post_id", "nunique"),
                total_engagement=("engagement", "sum"),
                    avg_sentiment=("post_sentiment", "mean"),
                    boost_score_sum=("boost_score", "sum"),
                    subreddit_diversity=("subreddit", "nunique"),
                weighted_numer=("weighted_numer", "sum"),
            )
        )
        
        # -- calculate the weighted sentiment.
            daily_df["weighted_sentiment"] = daily_df.apply(
                lambda r: (r["weighted_numer"] / r["total_engagement"]) if r["total_engagement"] else 0.0,
                axis=1,
            )

        # -- drop the weighted numer column.
        daily_df = daily_df.drop(columns=["weighted_numer"], errors="ignore")

        # -- rename the created_date column to date.
        daily_df = daily_df.rename(columns={"created_date": "date"}).sort_values(["date", "ticker"]).reset_index(drop=True)

        # -- update master daily table (append + dedupe by date,ticker)
        master_path = processed_reddit_by_day_dir / "reddit_daily_all.csv"
        
        try:
            # -- append to the master daily table if it exists.
            if master_path.exists():
                old = pd.read_csv(master_path)
                combined_all = pd.concat([old, daily_df], ignore_index=True)
            else:
                combined_all = daily_df.copy()
    
            # -- convert the date and ticker columns to strings.
            combined_all["date"] = combined_all["date"].astype(str)
            combined_all["ticker"] = combined_all["ticker"].astype(str)
            combined_all = (
                combined_all.drop_duplicates(subset=["date", "ticker"], keep="last")
                .sort_values(["date", "ticker"])
                .reset_index(drop=True)
            )

            # -- save the master daily table.
            combined_all.to_csv(master_path, index=False)
        except Exception:
            # non-fatal: don't break the pipeline if the master file is locked/corrupt
            pass

        return post_ticker_df


