# src/b_analysis/reddit_data_processor.py

"""
purpose:
  take Stage 1 CSV dumps, clean them, keep only financially relevant posts,
  attach tickers + sentiment, and emit a single Stage 2 dataset.

what this module does:
  - loads every `reddit_posts_*.csv` from `REDDIT_DATA_DIR`
  - dedupes by reddit id, cleans text, flags finance-ready posts
  - runs `TickerExtractor` for raw symbols + review diagnostics
  - calls `EntityLinker` to veto junk + boost confidence
  - scores sentiment via `SentimentScorer`
  - computes engagement + aggregates daily ticker metrics
  - writes processed CSV plus diagnostics (review queue, samples, sources)

how it plugs into other modules:
  - `TickerExtractor`: regex + context filtering to produce ticker candidates
  - `EntityLinker`: config-driven validation/boosting of those candidates
  - `SentimentScorer`: lexicon-based sentiment score for each post
  - `path_config`: tells us where raw, processed, and diagnostics files live
  - `config`: supplies ticker whitelist/blacklist/context keywords

what we get out:
  - `processed_reddit.csv` (clean posts with tickers, sentiment, engagement)
  - `ticker_daily_metrics.csv` (per-day aggregates)
  - `reddit_stage_metrics.csv`, `reddit_samples.json`,
    `ticker_review_queue.json`, `ticker_post_sources.json`
"""

# imports.
import json
import re
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from colorama import Fore, Style
from typing import List, Iterable, Optional

# local imports.
from src.b_analysis.entity_linker import EntityLinker
from src.b_analysis.ticker_extractor import TickerExtractor
from src.b_analysis.sentiment_scorer import SentimentScorer
from src.utils.path_config import (
    PROJECT_ROOT,
    RAW_REDDIT_DIR,
    PROCESSED_REDDIT_BY_DAY_DIR,
    PROCESSED_METRICS_DIR,
    DIAGNOSTICS_DIR,
)
from src.utils.config import (
    STOCK_DATA_BLACKLIST,
    WELL_KNOWN_TICKERS,
    VALID_ETFS,
    FINAL_STAGE_STOPWORDS,
    STRONG_FINANCE_WORDS,
    WEAK_FINANCE_WORDS,
)


class RedditDataProcessor:
    """process raw reddit data into a unified, cleaned dataset for topic identification."""

    # review allowed reasons.
    REVIEW_ALLOWED_REASONS = {
        "no_financial_context",
        "missing_context_keyword",
        "entity_linker_reject",  
        "timestamp_et",
    }

    # initialize reddit data processor.
    def __init__(self, input_files: Optional[Iterable[Path]] = None, run_date: Optional[str] = None, run_id: Optional[str] = None):
        print(f"{Fore.CYAN}initializing reddit data processor...{Style.RESET_ALL}")

        self._run_ts = datetime.now(timezone.utc)
        self.run_date = run_date or self._run_ts.date().isoformat()
        self.run_id = run_id or self._run_ts.strftime("%Y%m%d_%H%M%S")
        self.raw_count = 0

        self.input_files = [Path(p) for p in input_files] if input_files else []
        self.raw_root = RAW_REDDIT_DIR
        self.raw_day_dir = self._build_day_dir(self.raw_root, self.run_date)

        self.output_day_dir = self._build_day_dir(PROCESSED_REDDIT_BY_DAY_DIR, self.run_date)
        self.output_day_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_day_dir / f"processed_reddit_{self.run_date}.csv"
        self.review_queue_path = self.output_day_dir / f"ticker_review_queue_{self.run_date}.json"
        self.post_sources_path = self.output_day_dir / f"ticker_post_sources_{self.run_date}.json"
        self.metrics_path = PROCESSED_METRICS_DIR / "ticker_daily_metrics.csv"

        self.extractor = TickerExtractor()
        self.entity_linker = EntityLinker()
        self.sentiment_scorer = SentimentScorer()
        self.finance_keywords = {
            kw.lower() for kw in (STRONG_FINANCE_WORDS | WEAK_FINANCE_WORDS)
        }

        # optional seen-post registry hooks (disabled for now)
        self.enable_seen_registry = False
        self.seen_registry_path = PROCESSED_REDDIT_BY_DAY_DIR.parent / "seen_post_ids.json"
        self._seen_ids = self._load_seen_ids() if self.enable_seen_registry else set()

        print(f"{Fore.GREEN}✓ reddit data processor ready{Style.RESET_ALL}")

    @staticmethod
    def _build_day_dir(root: Path, run_date: str) -> Path:
        """Return YYYY/MM/DD folder under the given root."""
        try:
            year, month, day = run_date.split("-")
        except ValueError:
            raise ValueError(f"run_date must be YYYY-MM-DD, got {run_date}")
        return root / year / month / day

    def _load_seen_ids(self) -> set[str]:
        """Optional helper to load seen post ids (disabled by default)."""
        if not self.seen_registry_path.exists():
            return set()
        try:
            with open(self.seen_registry_path, "r", encoding="utf-8") as handle:
                return set(json.load(handle))
        except Exception:
            return set()

    def _persist_seen_ids(self) -> None:
        """Optional helper to persist seen ids (disabled by default)."""
        if not self.enable_seen_registry:
            return
        try:
            with open(self.seen_registry_path, "w", encoding="utf-8") as handle:
                json.dump(sorted(self._seen_ids), handle, indent=2)
        except Exception as exc:
            print(f"{Fore.YELLOW}warning: unable to persist seen ids ({exc}){Style.RESET_ALL}")

    @staticmethod
    def _format_path(path: Path) -> str:
        """Return a nice relative path when possible."""
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            try:
                return str(path.relative_to(Path.cwd()))
            except ValueError:
                return str(path)

    # ------------------------------------------------------------
    # cleaning
    # ------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        """clean text by removing urls, newlines, and extra whitespace."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[\n\r\t]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    # ------------------------------------------------------------
    # financial relevance heuristic
    # ------------------------------------------------------------
    def _is_financial_post(self, text: str) -> bool:
        """determine if a post is financial relevant based on keywords."""
        if not text:
            return False

        text_low = text.lower()
        return any(keyword in text_low for keyword in self.finance_keywords)

    # ------------------------------------------------------------
    # core processing
    # ------------------------------------------------------------
    def _build_daily_ticker_metrics(self, df: pd.DataFrame) -> None:
        """aggregate per-ticker metrics per day and persist to CSV."""
        
        if "created_utc" not in df.columns:
            return

        df = df.copy()
        df["date"] = pd.to_datetime(df["created_utc"]).dt.date

        # build daily ticker metrics.
        rows = []
        for _, row in df.iterrows():
            tickers = row.get("tickers", [])
            scores = row.get("ticker_scores", [])
            for ticker, score in zip(tickers, scores):
                rows.append({
                    "date": row["date"],
                    "ticker": ticker,
                    "confidence": score,
                    "engagement": row["engagement"],
                    "sentiment": row["sentiment"],
                    "is_relevant": row["is_relevant"],
                })

        if not rows:
            return

        # aggregate daily ticker metrics.
        daily = pd.DataFrame(rows)
        daily_agg = daily.groupby(["date", "ticker"]).agg({
            "confidence": "mean",
            "engagement": "sum",
            "sentiment": "mean",
            "is_relevant": "sum",
        }).reset_index()

        # save daily ticker metrics to CSV.
        output = self.metrics_path
        output.parent.mkdir(parents=True, exist_ok=True)
        daily_agg["run_date"] = self.run_date
        daily_agg["run_id"] = self.run_id
        header = not output.exists()
        daily_agg.to_csv(output, mode="a", header=header, index=False)
        print(f"{Fore.GREEN}✓ appended daily metrics → {self._format_path(output)}{Style.RESET_ALL}")

    # ------------------------------------------------------------

    def _write_stage_metrics(self, df: pd.DataFrame) -> None:
        """persist aggregate metrics for quick observability."""
        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

        # calculate average sentiment and confidence.
        avg_sentiment = float(df["sentiment"].mean()) if not df.empty else 0.0
        scores_series = (
            df["ticker_scores"].explode().dropna().astype(float)
            if not df.empty else pd.Series(dtype=float)
        )
        avg_confidence = float(scores_series.mean()) if not scores_series.empty else 0.0

        # build metrics dictionary.
        metrics = {
            "raw_posts": int(self.raw_count),
            "after_dedup": int(len(df)),
            "relevant_financial": int(df["is_relevant"].sum()) if "is_relevant" in df else 0,
            "with_tickers": int(df[df["tickers"].apply(len) > 0].shape[0]) if not df.empty else 0,
            "avg_sentiment": avg_sentiment,
            "avg_confidence": avg_confidence,
        }

        # save metrics to CSV.
        out = DIAGNOSTICS_DIR / "reddit_stage_metrics.csv"
        pd.DataFrame([metrics]).to_csv(out, index=False)
        print(f"{Fore.GREEN}✓ wrote stage metrics → {out}{Style.RESET_ALL}")

    # ------------------------------------------------------------
    def _write_debug_samples(self, df: pd.DataFrame) -> None:
        """store representative samples for manual QA."""
        if df.empty:
            return

        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

        # get top ticker posts, strong negative, strong positive, and random sample.
        top_ticker_posts = (
            df[df["tickers"].apply(len) > 0]
            .sort_values("engagement", ascending=False)
            .head(10)
        )
        strong_negative = df.nsmallest(min(20, len(df)), "sentiment")
        strong_positive = df.nlargest(min(20, len(df)), "sentiment")
        random_sample = df.sample(min(15, len(df)), random_state=42)

        # build samples dictionary.
        samples = {
            "top_ticker_posts": top_ticker_posts.to_dict(orient="records"),
            "strong_negative": strong_negative.to_dict(orient="records"),
            "strong_positive": strong_positive.to_dict(orient="records"),
            "random_sample": random_sample.to_dict(orient="records"),
        }

        # save samples to JSON.
        out = DIAGNOSTICS_DIR / "reddit_samples.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, default=str)

        print(f"{Fore.GREEN}✓ wrote debug samples → {self._format_path(out)}{Style.RESET_ALL}")

    # ------------------------------------------------------------
    def _write_ticker_review_queue(self, review_items: List[dict]) -> None:
        """persist rejected ticker candidates for manual config growth."""
        if not review_items:
            return

        out = self.review_queue_path
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            json.dump(review_items, f, indent=2, default=str)

        print(f"{Fore.GREEN}✓ wrote ticker review queue → {self._format_path(out)}{Style.RESET_ALL}")

    # ------------------------------------------------------------
    def _write_ticker_sources(self, df: pd.DataFrame, max_per_ticker: int = 5) -> None:
        """store sample posts for each accepted ticker."""
        if df.empty:
            return

        out = self.post_sources_path
        out.parent.mkdir(parents=True, exist_ok=True)

        # group posts by ticker.
        grouped = {}
        for _, row in df.iterrows():
            tickers = row.get("tickers", [])
            scores = row.get("ticker_scores", [])
            for ticker, confidence in zip(tickers, scores):
                entry = {
                    "post_id": row.get("id"),
                    "subreddit": row.get("subreddit"),
                    "score": row.get("score"),
                    "engagement": row.get("engagement"),
                    "sentiment": row.get("sentiment"),
                    "confidence": float(confidence),
                    "text_excerpt": (
                        row.get("cleaned_text", "")[:200]
                        if isinstance(row.get("cleaned_text"), str)
                        else ""
                    ),
                    "full_text": (
                        row.get("cleaned_text", "")
                        if isinstance(row.get("cleaned_text"), str)
                        else ""
                    ),
                    "title": row.get("title"),
                }
                grouped.setdefault(ticker, []).append(entry)

        if not grouped:
            return

        # build ticker samples dictionary.
        ticker_samples = {}
        for ticker, entries in grouped.items():
            # sort entries by engagement and confidence.
            sorted_entries = sorted(
                entries,
                key=lambda e: ((e.get("engagement") or 0), (e.get("confidence") or 0)),
                reverse=True,
            )
            ticker_samples[ticker] = sorted_entries[:max_per_ticker]

        # save ticker samples to JSON.
        with open(out, "w", encoding="utf-8") as f:
            json.dump(ticker_samples, f, indent=2, default=str)

        print(f"{Fore.GREEN}✓ wrote ticker post sources → {self._format_path(out)}{Style.RESET_ALL}")

    # ------------------------------------------------------------
    def _allow_final_ticker(self, ticker: str) -> bool:
        """final safeguard to prevent obvious false positives from leaving stage 2."""
        if not ticker:
            return False
        ticker_up = ticker.upper()
        if ticker_up in FINAL_STAGE_STOPWORDS:
            return False
        if ticker_up in STOCK_DATA_BLACKLIST:
            return False
        if ticker_up in WELL_KNOWN_TICKERS or ticker_up in VALID_ETFS:
            return True
        if len(ticker_up) <= 3:
            return False
        if not ticker_up.isalpha():
            return False
        return True

    # ------------------------------------------------------------
    def _log_summary(self, df: pd.DataFrame) -> None:
        """print concise summary stats to console."""
        ticker_rows = df[df["tickers"].apply(len) > 0] if not df.empty else pd.DataFrame()
        scores_series = (
            df["ticker_scores"].explode().astype(float)
            if not df.empty else pd.Series(dtype=float)
        )
        avg_confidence = scores_series.mean() if not scores_series.empty else 0.0
        avg_sentiment = df["sentiment"].mean() if not df.empty else 0.0

        print(
            f"""
            {Fore.MAGENTA}[Reddit Processor Summary]{Style.RESET_ALL}
            Raw posts:            {self.raw_count}
            After dedupe:         {len(df)}
            Relevant financial:   {int(df["is_relevant"].sum()) if "is_relevant" in df else 0}
            With tickers:         {len(ticker_rows)}
            Avg sentiment:        {avg_sentiment:.3f}
            Avg confidence:       {avg_confidence:.3f}
            """
        )

    # ------------------------------------------------------------
    def process(self) -> pd.DataFrame:
        """load run-specific raw reddit files → produce daily processed output."""
        raw_files = [path for path in self.input_files if Path(path).exists()]
        if not raw_files:
            raw_files = sorted(self.raw_day_dir.glob("reddit_posts_*.csv"))

        if not raw_files:
            print(f"{Fore.RED}✗ no raw reddit files found in {self.raw_day_dir}{Style.RESET_ALL}")
            return pd.DataFrame()

        dfs = []
        # read each raw reddit file.
        for f in raw_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"{Fore.RED}error reading {f}: {e}{Style.RESET_ALL}")

        if not dfs:
            return pd.DataFrame()

        # concatenate all raw reddit files.
        df = pd.concat(dfs, ignore_index=True)
        self.raw_count = len(df)
        if "id" in df.columns:
            df = df.drop_duplicates(subset="id")
        print(f"{Fore.GREEN}✓ loaded {len(df)} raw posts{Style.RESET_ALL}")

        text_col = "text" if "text" in df.columns else "selftext"

        # clean text.
        df["cleaned_text"] = df[text_col].fillna("") + " " + df["title"].fillna("")
        df["cleaned_text"] = df["cleaned_text"].apply(self._clean_text)

        # determine if the post is relevant.
        df["is_relevant"] = df["cleaned_text"].apply(self._is_financial_post)

        # score the sentiment of the post.
        df["sentiment"] = df["cleaned_text"].apply(self.sentiment_scorer.score)

        # calculate the engagement of the post.
        df["engagement"] = df["score"].fillna(0) + df["num_comments"].fillna(0)

        # extract tickers and confidence.
        tickers_list = []
        scores_list = []
        review_queue = []

        # iterate through each row of the dataframe.
        for _, row in df.iterrows():
            text = row.get("cleaned_text", "")
            tks, scs, review_items = self.extractor.extract_tickers(text)

            # convert scores to floats.
            scs = [float(s) if (s is not None and s != "") else 0.0 for s in scs]

            # boost the confidence of the tickers.
            tks, scs = self.entity_linker.boost_confidences(text, tks, scs)

            # filter the tickers and scores.
            filtered_pairs = [
                (ticker, score)
                for ticker, score in zip(tks, scs)
                if self._allow_final_ticker(ticker)
            ]
            if filtered_pairs:
                tks = [t for t, _ in filtered_pairs]
                scs = [s for _, s in filtered_pairs]
            else:
                tks, scs = [], []

            # append the tickers and scores to the lists.
            tickers_list.append(tks)
            scores_list.append(scs)

            # filter the review items.
            filtered_review = [
                item for item in review_items
                if item.get("reason") in self.REVIEW_ALLOWED_REASONS
            ]

            # build the base metadata.
            if filtered_review:
                base_meta = {
                    "post_id": row.get("id"),
                    "subreddit": row.get("subreddit"),
                    "score": row.get("score"),
                    "engagement": row.get("engagement"),
                    "text_excerpt": (text if isinstance(text, str) else ""),
                    "full_text": (text if isinstance(text, str) else ""),
                    "title": row.get("title"),
                }
                # append the enriched review items to the review queue.
                for item in filtered_review:
                    enriched = {**item, **base_meta}
                    review_queue.append(enriched)

        # append the tickers and scores to the dataframe.
        df["tickers"] = tickers_list
        df["ticker_scores"] = scores_list

        # only keep rows with at least 1 ticker.
        df = df[df["tickers"].apply(lambda x: len(x) > 0)]

        if df.empty:
            print(f"{Fore.YELLOW}no posts contained tickers. processed DF is empty.{Style.RESET_ALL}")
            return df

        # save the processed file.
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_file, index=False)

        # build per-day ticker metrics.
        self._build_daily_ticker_metrics(df)

        # write the stage metrics.
        self._write_stage_metrics(df)
        self._write_debug_samples(df)
        self._log_summary(df)
        self._write_ticker_review_queue(review_queue)
        self._write_ticker_sources(df)

        print(f"{Fore.GREEN}✓ saved processed reddit data → {self._format_path(self.output_file)}{Style.RESET_ALL}")
        return df