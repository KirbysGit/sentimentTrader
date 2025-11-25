# src/b_analysis/topic_identifier.py

"""
TopicIdentifier
---------------
Purpose:
    Identify trending tickers using already-processed Reddit data.

Assumptions:
    reddit_data_processor.py produces a processed_reddit.csv file with:
        - tickers: list of tickers per post
        - ticker_scores: ticker → confidence score dict
        - engagement: numeric engagement per post
        - date, subreddit, cleaned_text (optional but useful)

This module performs ONLY:
    - Aggregation
    - Scoring
    - Ranking

No ticker extraction.
No entity linking.
No complex validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
from colorama import Fore, Style

from src.utils.path_config import PROCESSED_REDDIT_DIR


class TopicIdentifier:
    def __init__(self):
        print(f"{Fore.CYAN}Initializing Topic Identifier...{Style.RESET_ALL}")

        self.processed_file = PROCESSED_REDDIT_DIR / "processed_reddit.csv"

        print(f"{Fore.GREEN}✓ Topic Identifier ready{Style.RESET_ALL}")

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def _load_processed_df(self) -> pd.DataFrame:
        """Load processed Reddit data."""
        if not self.processed_file.exists():
            print(f"{Fore.RED}✗ No processed_reddit.csv found at {self.processed_file}{Style.RESET_ALL}")
            return pd.DataFrame()

        df = pd.read_csv(self.processed_file)

        # Convert tickers and ticker_scores from strings → Python objects
        if "tickers" in df.columns:
            df["tickers"] = df["tickers"].apply(self._safe_eval_list)

        if "ticker_scores" in df.columns:
            df["ticker_scores"] = df["ticker_scores"].apply(self._safe_eval_dict)

        return df

    def _safe_eval_list(self, x):
        """Convert stringified list to list safely."""
        if isinstance(x, list):
            return x
        try:
            return eval(x) if isinstance(x, str) else []
        except:
            return []

    def _safe_eval_dict(self, x):
        """Convert stringified dict to dict safely."""
        if isinstance(x, dict):
            return x
        try:
            return eval(x) if isinstance(x, str) else {}
        except:
            return {}

    # ------------------------------------------------------------
    # Trending logic
    # ------------------------------------------------------------
    def _expand_tickers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide-format rows:
            tickers = ["NVDA", "AMD"]
            ticker_scores = {"NVDA": 0.9, "AMD": 0.4}
        Into long-format rows:
            ticker, confidence, engagement, date, subreddit
        """
        rows = []

        for _, row in df.iterrows():
            tickers = row.get("tickers", [])
            scores = row.get("ticker_scores", {})
            engagement = row.get("engagement", 0)
            date = row.get("date", None)
            subreddit = row.get("subreddit", None)
            context = row.get("cleaned_text", "")[:200]

            for ticker in tickers:
                conf = scores.get(ticker, 0)

                rows.append({
                    "ticker": ticker,
                    "confidence": conf,
                    "engagement": engagement,
                    "date": date,
                    "subreddit": subreddit,
                    "context": context,
                })

        return pd.DataFrame(rows)

    def _score_tickers(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """Aggregate and score each ticker."""

        grouped = df_long.groupby("ticker").agg({
            "confidence": "mean",
            "engagement": "sum",
            "ticker": "count",
        }).rename(columns={"ticker": "mentions"})

        # Final score:
        #   mentions * avg_confidence * log(1 + engagement)
        grouped["score"] = (
            grouped["mentions"]
            * grouped["confidence"]
            * np.log1p(grouped["engagement"])
        )

        return grouped.sort_values("score", ascending=False).reset_index()

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def identify_trending(self, top_n: int = 10) -> pd.DataFrame:
        """Main entry: returns top trending tickers."""

        df = self._load_processed_df()
        if df.empty:
            return pd.DataFrame()

        df_long = self._expand_tickers(df)
        if df_long.empty:
            print(f"{Fore.YELLOW}No ticker data available in processed file.{Style.RESET_ALL}")
            return pd.DataFrame()

        scoring = self._score_tickers(df_long)

        # Save trending CSV
        from src.utils.path_config import TICKER_GENERAL_DIR
        out_file = TICKER_GENERAL_DIR / f"trending_{df['date'].max()}.csv"
        scoring.to_csv(out_file, index=False)
        print(f"{Fore.GREEN}✓ Saved trending ticker analysis to {out_file}{Style.RESET_ALL}")

        # Return top N
        return scoring.head(top_n)

    def get_trending_tickers(self, top_n: int = 10) -> List[str]:
        """Return just the ticker symbols."""
        df = self.identify_trending(top_n=top_n)
        return df["ticker"].tolist() if not df.empty else []

