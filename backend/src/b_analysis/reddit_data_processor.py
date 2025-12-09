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
import numpy as np
from colorama import Fore, Style
from typing import List, Iterable, Optional

# local imports.
from src.b_analysis.entity_linker import EntityLinker
from src.b_analysis.ticker_extractor import TickerExtractor
from src.b_analysis.sentiment_scorer import SentimentScorer
from src.b_analysis.confidence_scorer import ConfidenceScorer
from src.utils.path_config import (
    PROJECT_ROOT,
    RAW_REDDIT_DIR,
    PROCESSED_REDDIT_BY_DAY_DIR,
    PROCESSED_METRICS_DIR,
    DIAGNOSTICS_DIR,
)
from src.utils.config import (
    BLOCKLIST,
    WELL_KNOWN_TICKERS,
    VALID_ETFS,
    ALWAYS_ALLOW,
    FINAL_STAGE_STOPWORDS,
    FINANCE_CONTEXT_WORDS,
)
from src.utils.ticker_aliases import get_canonical_alias_map


class RedditDataProcessor:
    """process raw reddit data into a unified, cleaned dataset for topic identification."""

    # review allowed reasons.
    REVIEW_ALLOWED_REASONS = {
        "no_financial_context",
        "missing_context_keyword",
        "entity_linker_reject",  
        "timestamp_et",
        "unknown_symbol",
        "negative_context",
    }

    # minimal feature columns that move forward to Stage 3 / feature builder.
    PROCESSED_OUTPUT_COLUMNS = [
        "id",
        "created_utc",
        "subreddit",
        "flair",
        "author_name",
        "author_karma",
        "author_is_mod",
        "author_created_utc",
        "score",
        "num_comments",
        "upvote_ratio",
        "engagement",
        "is_relevant",
        "sentiment",
        "tickers",
        "ticker_scores",
        "ticker_confidence_reasons",
        "ticker_in_title",
        "num_tickers_in_post",
        "is_portfolio_post",
        "is_watchlist_post",
        "has_position_language",
        "run_date",
        "run_id",
    ]

    PORTFOLIO_KEYWORDS = (
        "my portfolio",
        "portfolio:",
        "current portfolio",
        "portfolio allocation",
        "allocation",
        "allocated",
        "allocations",
        "my holdings",
        "holdings:",
        "my positions",
        "positions:",
        "exposure",
        "weighting",
        "weights",
    )

    WATCHLIST_KEYWORDS = (
        "watchlist",
        "watch list",
        "list of tickers",
        "ticker list",
        "shopping list",
        "ideas list",
        "top tickers",
        "ticker summary",
    )

    STRONG_POSITION_PATTERNS = (
        r"\bi(?:\s*(am|’m|'m|m))?\s+long\b",
        r"\bi(?:\s*(am|’m|'m|m))?\s+short\b",
        r"\bi\s+(bought|bought in|loaded|went in|got in)\b",
        r"\bi\s+(sold|dumped|trimmed|got out of)\b",
        r"\bmy\s+(position|positions|shares|calls|puts)\b",
        r"\bstill\s+holding\b",
        r"\bbagholding\b",
        r"\bi\s+hold\b",
        r"\bi\s+own\b",
        r"\bi\s+added\b",
        r"\bi\s+trimmed\b",
        r"\bi\s+took\s+profit\b",
    )

    WEAK_POSITION_PATTERNS = (
        r"\bbet(?:ting)?\s+on\b",
        r"\bplay(?:ing)?\b",
        r"\blooking\s+to\s+(buy|sell)\b",
        r"\bthinking\s+of\s+(buying|selling)\b",
        r"\bmoney\s+to\s+be\s+made\b",
    )

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
        self.confidence_scorer = ConfidenceScorer()
        self.finance_keywords = {kw.lower() for kw in FINANCE_CONTEXT_WORDS}
        self.canonical_alias_map = get_canonical_alias_map()

        # optional seen-post registry hooks (disabled for now)
        self.enable_seen_registry = True
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

    def _record_seen_ids(self, df: pd.DataFrame) -> None:
        """Update seen-id registry with posts processed this run."""
        if not self.enable_seen_registry or "id" not in df.columns or df.empty:
            return
        ids_series = df["id"].dropna().astype(str)
        new_ids = {i for i in ids_series if i and i.lower() != "nan"}
        if not new_ids:
            return
        added = new_ids - self._seen_ids
        if not added:
            return
        self._seen_ids.update(added)
        self._persist_seen_ids()

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

    def _canonicalize_ticker(self, ticker: str) -> str:
        """normalize a ticker using the alias map, defaulting to uppercase original."""
        if ticker is None:
            return ""
        if isinstance(ticker, float) and np.isnan(ticker):
            return ""
        ticker_str = str(ticker).strip()
        if not ticker_str:
            return ""
        ticker_up = ticker_str.upper()
        return self.canonical_alias_map.get(ticker_up, ticker_up)

    @staticmethod
    def _build_ticker_snippet(text: str, ticker: str, radius: int = 30) -> str:
        """Return a short excerpt centered on the first occurrence of ticker."""
        if not isinstance(text, str) or not ticker:
            return (text or "")[: radius * 2]
        text_low = text.lower()
        ticker_low = ticker.lower()
        idx = text_low.find(ticker_low)
        if idx == -1:
            return text[: radius * 2]
        start = max(0, idx - radius)
        end = min(len(text), idx + len(ticker) + radius)
        return text[start:end].strip()

    def _has_position_language(self, text: str) -> bool:
        """Detect whether the post discusses specific trade positioning."""
        if not text:
            return False
        text_low = text.lower()
        for pattern in self.STRONG_POSITION_PATTERNS:
            if re.search(pattern, text_low):
                return True
        # weak signals require first-person ownership context
        if not re.search(r"\b(i|my|we|our|me|us)\b", text_low):
            return False
        for pattern in self.WEAK_POSITION_PATTERNS:
            if re.search(pattern, text_low):
                return True
        return False

    def _is_portfolio_post(self, text: str, tickers: List[str]) -> bool:
        """Heuristic to detect allocation / portfolio breakdown posts."""
        if not text:
            return False
        text_low = text.lower()
        num_tickers = len(tickers or [])
        if num_tickers >= 3 and "%" in text:
            return True
        if num_tickers >= 2 and any(keyword in text_low for keyword in self.PORTFOLIO_KEYWORDS):
            return True
        return False

    def _is_watchlist_post(self, text: str, tickers: List[str], has_position_language: bool) -> bool:
        """Heuristic to identify list/watchlist style posts."""
        if not text:
            return False
        text_low = text.lower()
        if any(keyword in text_low for keyword in self.WATCHLIST_KEYWORDS):
            return True
        bullet_hits = len(re.findall(r"(?:^|\n)[\-\*\u2022]\s*\$?[A-Za-z]{2,5}\b", text, flags=re.MULTILINE))
        if bullet_hits >= 3:
            return True
        if len(tickers) >= 5 and not has_position_language:
            return True
        return False

    @staticmethod
    def _ticker_in_title_flags(title: str, tickers: List[str]) -> List[bool]:
        """Return per-ticker booleans indicating presence in the title."""
        if not isinstance(title, str):
            title = ""
        title_upper = title.upper()
        flags: List[bool] = []
        for ticker in tickers:
            symbol = ticker.upper() if ticker else ""
            if not symbol:
                flags.append(False)
                continue
            pattern = rf"(?:^|[^A-Z0-9\$])\$?{re.escape(symbol)}(?:[^A-Z0-9]|$)"
            flags.append(bool(re.search(pattern, title_upper)))
        return flags

    @staticmethod
    def _compute_ticker_weight(
        num_tickers: int,
        in_title: bool,
        is_portfolio_post: bool,
        is_watchlist_post: bool,
        has_position_language: bool,
    ) -> float:
        """Derive a per-mention weight to scale sentiment/engagement impact."""
        weight = 1.0
        num_tickers = max(1, num_tickers or 1)

        if num_tickers >= 6:
            weight *= 0.5
        elif num_tickers == 1:
            weight *= 1.2

        if is_portfolio_post:
            weight *= 0.3
        if is_watchlist_post:
            weight *= 0.6
        if in_title:
            weight *= 1.5
        if has_position_language:
            weight *= 2.0

        return float(min(max(weight, 0.05), 5.0))

    def _build_processed_output_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """trim processed dataframe down to feature-ready columns."""
        df = df.copy()
        df["run_date"] = self.run_date
        df["run_id"] = self.run_id
        columns = [col for col in self.PROCESSED_OUTPUT_COLUMNS if col in df.columns]
        return df[columns]

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
        """aggregate per-ticker metrics per day and upsert into the master CSV."""

        if "created_utc" not in df.columns:
            return

        df = df.copy()
        df["date"] = pd.to_datetime(df["created_utc"]).dt.date

        rows = []
        for _, row in df.iterrows():
            tickers = row.get("tickers", []) or []
            scores = row.get("ticker_scores", []) or []
            if not tickers or not scores:
                continue

            engagement_val = row.get("engagement", 0.0)
            sentiment_val = row.get("sentiment", 0.0)
            engagement = float(engagement_val) if pd.notna(engagement_val) else 0.0
            sentiment = float(sentiment_val) if pd.notna(sentiment_val) else 0.0
            is_relevant = bool(row.get("is_relevant", False))
            title_flags = row.get("ticker_in_title", []) or []
            num_tickers = int(row.get("num_tickers_in_post") or len(tickers) or 0)
            if num_tickers <= 0:
                num_tickers = len(tickers)
            is_portfolio_post = bool(row.get("is_portfolio_post", False))
            is_watchlist_post = bool(row.get("is_watchlist_post", False))
            has_position_language = bool(row.get("has_position_language", False))
            if len(tickers) != len(scores):
                min_len = min(len(tickers), len(scores))
                tickers = tickers[:min_len]
                scores = scores[:min_len]
                title_flags = title_flags[:min_len] if len(title_flags) >= min_len else title_flags
            if len(title_flags) > len(tickers):
                title_flags = title_flags[: len(tickers)]

            for idx, (ticker, score) in enumerate(zip(tickers, scores)):
                in_title = bool(title_flags[idx]) if idx < len(title_flags) else False
                weight = self._compute_ticker_weight(
                    num_tickers=num_tickers,
                    in_title=in_title,
                    is_portfolio_post=is_portfolio_post,
                    is_watchlist_post=is_watchlist_post,
                    has_position_language=has_position_language,
                )
                rows.append(
                    {
                        "date": row["date"],
                        "ticker": ticker,
                        "confidence": float(score),
                        "engagement": engagement,
                        "sentiment": sentiment,
                        "is_relevant": is_relevant,
                        "weight": weight,
                        "in_title": in_title,
                        "has_position_language": has_position_language,
                        "is_portfolio_post": is_portfolio_post,
                        "is_watchlist_post": is_watchlist_post,
                        "num_tickers_in_post": num_tickers,
                        "weighted_engagement": engagement * weight,
                        "weighted_sentiment": sentiment * weight,
                        "weighted_sentiment_eng": sentiment * engagement * weight,
                    }
                )

        if not rows:
            return

        daily = pd.DataFrame(rows)
        daily["ticker"] = daily["ticker"].apply(self._canonicalize_ticker)
        daily = daily[daily["ticker"] != ""]
        daily["is_relevant_int"] = daily["is_relevant"].astype(int)
        daily["sentiment_x_eng"] = daily["sentiment"] * daily["engagement"]
        daily["has_position_language_int"] = daily["has_position_language"].astype(int)
        daily["in_title_int"] = daily["in_title"].astype(int)

        grouped = daily.groupby(["date", "ticker"], as_index=False)
        daily_agg = grouped.agg(
            num_mentions=("sentiment", "size"),
            mean_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
            total_engagement=("engagement", "sum"),
            mean_engagement=("engagement", "mean"),
            mean_sentiment=("sentiment", "mean"),
            sentiment_std=("sentiment", "std"),
            sentiment_x_eng_sum=("sentiment_x_eng", "sum"),
            relevant_count=("is_relevant_int", "sum"),
            total_weight=("weight", "sum"),
            weighted_engagement=("weighted_engagement", "sum"),
            weighted_sentiment_sum=("weighted_sentiment", "sum"),
            weighted_sentiment_eng=("weighted_sentiment_eng", "sum"),
            strong_post_count=("has_position_language_int", "sum"),
            title_focus_mentions=("in_title_int", "sum"),
        )

        daily_agg["sentiment_std"] = daily_agg["sentiment_std"].fillna(0.0)
        daily_agg["eng_weighted_sentiment"] = daily_agg["sentiment_x_eng_sum"] / daily_agg[
            "total_engagement"
        ].replace(0, np.nan)
        daily_agg["eng_weighted_sentiment"] = daily_agg["eng_weighted_sentiment"].fillna(0.0)
        daily_agg["weighted_sentiment_avg"] = (
            daily_agg["weighted_sentiment_sum"] / daily_agg["total_weight"].replace(0, np.nan)
        )
        daily_agg["weighted_sentiment_avg"] = daily_agg["weighted_sentiment_avg"].fillna(0.0)
        daily_agg["weighted_engagement_sentiment"] = (
            daily_agg["weighted_sentiment_eng"] / daily_agg["weighted_engagement"].replace(0, np.nan)
        )
        daily_agg["weighted_engagement_sentiment"] = daily_agg["weighted_engagement_sentiment"].fillna(0.0)
        daily_agg["relevant_ratio"] = daily_agg["relevant_count"] / daily_agg["num_mentions"].clip(
            lower=1
        )
        daily_agg["log_total_engagement"] = np.log1p(
            daily_agg["total_engagement"].clip(lower=0.0)
        )
        daily_agg["log_num_mentions"] = np.log1p(daily_agg["num_mentions"].clip(lower=0))
        daily_agg["run_date"] = getattr(self, "run_date", None)
        daily_agg["run_id"] = getattr(self, "run_id", None)
        daily_agg = daily_agg.drop(columns=["sentiment_x_eng_sum"])

        output = self.metrics_path
        output.parent.mkdir(parents=True, exist_ok=True)

        if output.exists():
            try:
                existing = pd.read_csv(output)
            except Exception:
                existing = pd.DataFrame()
        else:
            existing = pd.DataFrame()

        if not existing.empty and "ticker" in existing.columns:
            existing["ticker"] = existing["ticker"].apply(self._canonicalize_ticker)
            existing = existing[existing["ticker"] != ""]

        combined = pd.concat([existing, daily_agg], ignore_index=True)
        if not combined.empty:
            combined["date"] = pd.to_datetime(combined["date"]).dt.date
            combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
            combined = combined.sort_values(["date", "ticker"])

        combined.to_csv(output, index=False)
        print(f"{Fore.GREEN}✓ wrote daily metrics → {self._format_path(output)}{Style.RESET_ALL}")

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

        # de-duplicate review entries to avoid repeat noise across runs.
        # use post_id + ticker + reason as a stable key.
        seen_keys = set()
        deduped: List[dict] = []
        prior_seen_ids = self._seen_ids if self.enable_seen_registry else set()
        for item in review_items:
            post_id = item.get("post_id")
            if post_id and post_id in prior_seen_ids:
                continue  # already processed in a previous run
            key = (post_id, item.get("ticker"), item.get("reason"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)

        if not deduped:
            return

        out = self.review_queue_path
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=2, default=str)

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
            reasons = row.get("ticker_confidence_reasons", [])
            title_flags = row.get("ticker_in_title", [])
            for idx, (ticker, confidence) in enumerate(zip(tickers, scores)):
                reason = reasons[idx] if isinstance(reasons, list) and idx < len(reasons) else ""
                text = row.get("cleaned_text", "")
                entry = {
                    "post_id": row.get("id"),
                    "subreddit": row.get("subreddit"),
                    "score": row.get("score"),
                    "engagement": row.get("engagement"),
                    "sentiment": row.get("sentiment"),
                    "confidence": float(confidence),
                    "confidence_reason": reason,
                    "context_flags": {
                        "in_title": bool(title_flags[idx]) if isinstance(title_flags, list) and idx < len(title_flags) else False,
                        "num_tickers_in_post": int(row.get("num_tickers_in_post", 0) or 0),
                        "is_portfolio_post": bool(row.get("is_portfolio_post", False)),
                        "is_watchlist_post": bool(row.get("is_watchlist_post", False)),
                        "has_position_language": bool(row.get("has_position_language", False)),
                    },
                    "text_excerpt": self._build_ticker_snippet(
                        text if isinstance(text, str) else "",
                        ticker,
                        radius=30,
                    ),
                    "full_text": (
                        text
                        if isinstance(text, str)
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
        if ticker_up in BLOCKLIST:
            return False
        if ticker_up in WELL_KNOWN_TICKERS or ticker_up in ALWAYS_ALLOW or ticker_up in VALID_ETFS:
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
            df["id"] = df["id"].astype(str)
            df = df.drop_duplicates(subset="id")
            if self.enable_seen_registry and self._seen_ids:
                before_seen_filter = len(df)
                df = df[~df["id"].isin(self._seen_ids)]
                skipped = before_seen_filter - len(df)
                if skipped > 0:
                    print(
                        f"{Fore.YELLOW}⟳ skipped {skipped} previously processed posts (seen registry){Style.RESET_ALL}"
                    )
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
        confidence_reasons_list = []
        ticker_in_title_flags_list: List[List[bool]] = []
        num_ticker_counts: List[int] = []
        portfolio_post_flags: List[bool] = []
        watchlist_post_flags: List[bool] = []
        position_language_flags: List[bool] = []
        review_queue = []

        # iterate through each row of the dataframe.
        for _, row in df.iterrows():
            text = row.get("cleaned_text", "")
            title_text = row.get("title") or ""
            body_raw = row.get(text_col, "")
            raw_context_text = f"{title_text}\n{body_raw}" if isinstance(body_raw, str) else str(title_text or "")
            tks, base_scores, review_items, evidence_items = self.extractor.extract_tickers(text)
            base_scores = [float(s) if (s is not None and s != "") else 0.0 for s in base_scores]

            final_scores: List[float] = []
            confidence_notes: List[str] = []
            if tks:
                for ticker, base_score, evidence in zip(tks, base_scores, evidence_items):
                    scored_value, summary = self.confidence_scorer.score(ticker, base_score, evidence)
                    final_scores.append(scored_value)
                    confidence_notes.append(summary)

            approved_tickers: List[str] = []
            approved_scores: List[float] = []
            approved_notes: List[str] = []
            for ticker, score, note in zip(tks, final_scores, confidence_notes):
                if not self._allow_final_ticker(ticker):
                    continue
                canonical = self._canonicalize_ticker(ticker)
                approved_tickers.append(canonical)
                approved_scores.append(score)
                approved_notes.append(note)

            if approved_tickers:
                tks = approved_tickers
                scs = approved_scores
                note_sequence = approved_notes
            else:
                tks, scs = [], []
                note_sequence = []

            has_position_language = self._has_position_language(raw_context_text)
            is_portfolio_post = self._is_portfolio_post(raw_context_text, tks)
            is_watchlist_post = self._is_watchlist_post(raw_context_text, tks, has_position_language)
            title_flags = self._ticker_in_title_flags(title_text, tks)
            num_tickers = len(tks)

            # append the tickers and scores to the lists.
            tickers_list.append(tks)
            scores_list.append(scs)
            confidence_reasons_list.append(note_sequence)
            ticker_in_title_flags_list.append(title_flags)
            num_ticker_counts.append(num_tickers)
            portfolio_post_flags.append(is_portfolio_post)
            watchlist_post_flags.append(is_watchlist_post)
            position_language_flags.append(has_position_language)

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
                    "full_text": (text if isinstance(text, str) else ""),
                    "title": row.get("title"),
                }
                # append the enriched review items to the review queue.
                for item in filtered_review:
                    snippet = self._build_ticker_snippet(
                        text if isinstance(text, str) else "",
                        item.get("ticker", ""),
                        radius=30,
                    )
                    enriched = {**item, **base_meta, "text_excerpt": snippet}
                    review_queue.append(enriched)

        # append the tickers and scores to the dataframe.
        df["tickers"] = tickers_list
        df["ticker_scores"] = scores_list
        df["ticker_confidence_reasons"] = confidence_reasons_list
        df["ticker_in_title"] = ticker_in_title_flags_list
        df["num_tickers_in_post"] = num_ticker_counts
        df["is_portfolio_post"] = portfolio_post_flags
        df["is_watchlist_post"] = watchlist_post_flags
        df["has_position_language"] = position_language_flags

        # only keep rows with at least 1 ticker.
        df = df[df["tickers"].apply(lambda x: len(x) > 0)]

        if df.empty:
            print(f"{Fore.YELLOW}no posts contained tickers. processed DF is empty.{Style.RESET_ALL}")
            return df

        # save the processed file.
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        output_df = self._build_processed_output_df(df)
        output_df.to_csv(self.output_file, index=False)

        # build per-day ticker metrics.
        self._build_daily_ticker_metrics(df)

        # write the stage metrics.
        self._write_stage_metrics(df)
        self._write_debug_samples(df)
        self._log_summary(df)
        self._write_ticker_review_queue(review_queue)
        self._write_ticker_sources(df)
        self._record_seen_ids(df)

        print(f"{Fore.GREEN}✓ saved processed reddit data → {self._format_path(self.output_file)}{Style.RESET_ALL}")
        return df