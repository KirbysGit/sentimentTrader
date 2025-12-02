# src/b_analysis/reddit_data_processor.py

"""
RedditDataProcessor
-------------------
Consumes raw Reddit CSVs from the collector and produces a single file:
    processed_reddit.csv

Responsibilities:
    - Load raw posts
    - Clean + normalize text
    - Determine financial relevance
    - Extract tickers with TickerExtractor
    - Score ticker confidence
    - Compute engagement score
    - Save unified processed dataset for TopicIdentifier
"""

import json
import re
import pandas as pd
from colorama import Fore, Style

from src.b_analysis.ticker_extractor import TickerExtractor
from src.b_analysis.entity_linker import EntityLinker  # minimal version
from src.b_analysis.sentiment_scorer import SentimentScorer
from src.utils.path_config import REDDIT_DATA_DIR, PROCESSED_REDDIT_DIR, DIAGNOSTICS_DIR


class RedditDataProcessor:
    def __init__(self):
        print(f"{Fore.CYAN}Initializing Reddit Data Processor...{Style.RESET_ALL}")

        self.raw_dir = REDDIT_DATA_DIR
        self.output_file = PROCESSED_REDDIT_DIR / "processed_reddit.csv"
        self.raw_count = 0

        # Modules
        self.extractor = TickerExtractor()
        self.entity_linker = EntityLinker()
        self.sentiment_scorer = SentimentScorer()

        print(f"{Fore.GREEN}✓ Reddit Data Processor ready{Style.RESET_ALL}")

    # ------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[\n\r\t]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    # ------------------------------------------------------------
    # Financial relevance heuristic
    # ------------------------------------------------------------
    def _is_financial_post(self, text: str) -> bool:
        if not text:
            return False

        keywords = [
            "stock", "market", "earnings", "revenue", "growth",
            "share", "trade", "price", "calls", "puts", "options",
            "bull", "bear", "ETF", "invest", "buy", "sell",
            "$", "fed", "inflation"
        ]
        return any(word.lower() in text.lower() for word in keywords)

    # ------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------
    def _build_daily_ticker_metrics(self, df: pd.DataFrame) -> None:
        """Aggregate per-ticker metrics per day and persist to CSV."""
        if "created_utc" not in df.columns:
            return

        df = df.copy()
        df["date"] = pd.to_datetime(df["created_utc"]).dt.date

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

        daily = pd.DataFrame(rows)
        daily_agg = daily.groupby(["date", "ticker"]).agg({
            "confidence": "mean",
            "engagement": "sum",
            "sentiment": "mean",
            "is_relevant": "sum",
        }).reset_index()

        output = PROCESSED_REDDIT_DIR / "ticker_daily_metrics.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        daily_agg.to_csv(output, index=False)
        print(f"{Fore.GREEN}✓ Saved → {output}{Style.RESET_ALL}")

    # ------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------
    def _write_stage_metrics(self, df: pd.DataFrame) -> None:
        """Persist aggregate metrics for quick observability."""
        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

        avg_sentiment = float(df["sentiment"].mean()) if not df.empty else 0.0
        scores_series = (
            df["ticker_scores"].explode().dropna().astype(float)
            if not df.empty else pd.Series(dtype=float)
        )
        avg_confidence = float(scores_series.mean()) if not scores_series.empty else 0.0

        metrics = {
            "raw_posts": int(self.raw_count),
            "after_dedup": int(len(df)),
            "relevant_financial": int(df["is_relevant"].sum()) if "is_relevant" in df else 0,
            "with_tickers": int(df[df["tickers"].apply(len) > 0].shape[0]) if not df.empty else 0,
            "avg_sentiment": avg_sentiment,
            "avg_confidence": avg_confidence,
        }

        out = DIAGNOSTICS_DIR / "reddit_stage_metrics.csv"
        pd.DataFrame([metrics]).to_csv(out, index=False)
        print(f"{Fore.GREEN}✓ Wrote stage metrics → {out}{Style.RESET_ALL}")

    # ------------------------------------------------------------
    def _write_debug_samples(self, df: pd.DataFrame) -> None:
        """Store representative samples for manual QA."""
        if df.empty:
            return

        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

        top_ticker_posts = (
            df[df["tickers"].apply(len) > 0]
            .sort_values("engagement", ascending=False)
            .head(10)
        )
        strong_negative = df.nsmallest(min(20, len(df)), "sentiment")
        strong_positive = df.nlargest(min(20, len(df)), "sentiment")
        random_sample = df.sample(min(15, len(df)), random_state=42)

        samples = {
            "top_ticker_posts": top_ticker_posts.to_dict(orient="records"),
            "strong_negative": strong_negative.to_dict(orient="records"),
            "strong_positive": strong_positive.to_dict(orient="records"),
            "random_sample": random_sample.to_dict(orient="records"),
        }

        out = DIAGNOSTICS_DIR / "reddit_samples.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, default=str)

        print(f"{Fore.GREEN}✓ Wrote debug samples → {out}{Style.RESET_ALL}")

    # ------------------------------------------------------------
    def _log_summary(self, df: pd.DataFrame) -> None:
        """Print concise summary stats to console."""
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
    # Core processing
    # ------------------------------------------------------------
    def process(self) -> pd.DataFrame:
        """Load all raw reddit files → produce processed_reddit.csv."""
        raw_files = sorted(self.raw_dir.glob("reddit_posts_*.csv"))
        if not raw_files:
            print(f"{Fore.RED}✗ No raw reddit files found in {self.raw_dir}{Style.RESET_ALL}")
            return pd.DataFrame()

        dfs = []
        for f in raw_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"{Fore.RED}Error reading {f}: {e}{Style.RESET_ALL}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        self.raw_count = len(df)
        if "id" in df.columns:
            df = df.drop_duplicates(subset="id")
        print(f"{Fore.GREEN}✓ Loaded {len(df)} raw posts{Style.RESET_ALL}")

        text_col = "text" if "text" in df.columns else "selftext"

        df["cleaned_text"] = df[text_col].fillna("") + " " + df["title"].fillna("")
        df["cleaned_text"] = df["cleaned_text"].apply(self._clean_text)

        # Relevance
        df["is_relevant"] = df["cleaned_text"].apply(self._is_financial_post)

        # Sentiment
        df["sentiment"] = df["cleaned_text"].apply(self.sentiment_scorer.score)

        # Engagement
        df["engagement"] = df["score"].fillna(0) + df["num_comments"].fillna(0)

        # Extract tickers + confidence
        tickers_list = []
        scores_list = []

        for text in df["cleaned_text"]:
            tks, scs = self.extractor.extract_tickers(text)

            # force float conversion.
            scs = [float(s) if (s is not None and s != "") else 0.0 for s in scs]

            # now safe to boost.
            tks, scs = self.entity_linker.boost_confidences(text, tks, scs)


            tickers_list.append(tks)
            scores_list.append(scs)

        df["tickers"] = tickers_list
        df["ticker_scores"] = scores_list

        # Only keep rows with at least 1 ticker
        df = df[df["tickers"].apply(lambda x: len(x) > 0)]

        if df.empty:
            print(f"{Fore.YELLOW}No posts contained tickers. Processed DF is empty.{Style.RESET_ALL}")
            return df

        # Save processed file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_file, index=False)

        # Build per-day ticker metrics
        self._build_daily_ticker_metrics(df)

        # Diagnostics & observability
        self._write_stage_metrics(df)
        self._write_debug_samples(df)
        self._log_summary(df)

        print(f"{Fore.GREEN}✓ Saved processed reddit data → {self.output_file}{Style.RESET_ALL}")
        return df


# ------------------------------------------------------------
# Direct test run
# ------------------------------------------------------------
if __name__ == "__main__":
    processor = RedditDataProcessor()
    processor.process()
