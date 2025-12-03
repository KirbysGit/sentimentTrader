# src/b_analysis/topic_identifier.py

"""
purpose:
  take the stage 2 output (`processed_reddit.csv`) and surface
  the highest-confidence, highest-engagement tickers currently trending.

what this module does:
  - loads the processed file from `PROCESSED_REDDIT_DIR`
  - expands each post into per-ticker rows (ticker, confidence, engagement, date)
  - aggregates mentions + engagement + avg confidence
  - computes a composite score (mentions * avg_confidence * log(1+engagement))
  - saves a ranked CSV under `TICKER_GENERAL_DIR`

how it fits:
  stage 3 "analysis/insights". stage 1 collects raw posts → stage 2 cleans +
  attaches tickers → TopicIdentifier ranks which tickers deserve attention.
  No extraction or sentiment logic lives here—only aggregation + ranking.

expansion ideas:
  - add sector grouping or subreddit breakdown columns
  - include momentum calculations (day-over-day deltas)
  - enrich scoring with external fundamentals or price moves
"""

# imports.
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
from colorama import Fore, Style

# local imports.
from src.utils.path_config import PROCESSED_REDDIT_BY_DAY_DIR


class TopicIdentifier:
    def __init__(self, target_date: Optional[str] = None):
        print(f"{Fore.CYAN}initializing topic identifier...{Style.RESET_ALL}")

        self.by_day_root = PROCESSED_REDDIT_BY_DAY_DIR
        self.processed_file = self._resolve_processed_path(target_date)

        print(f"{Fore.GREEN}✓ topic identifier ready{Style.RESET_ALL}")

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def _load_processed_df(self) -> pd.DataFrame:
        """load processed Reddit data."""
        if not self.processed_file.exists():
            print(f"{Fore.RED}✗ no processed_reddit.csv found at {self.processed_file}{Style.RESET_ALL}")
            return pd.DataFrame()

        df = pd.read_csv(self.processed_file)

        # convert tickers and ticker_scores from strings → Python objects
        if "tickers" in df.columns:
            df["tickers"] = df["tickers"].apply(self._safe_eval_list)

        if "ticker_scores" in df.columns:
            df["ticker_scores"] = df["ticker_scores"].apply(self._safe_eval_dict)

        return df

    def _resolve_processed_path(self, target_date: Optional[str]) -> Path:
        """Pick the correct processed_reddit file based on the target date (latest if None)."""
        if target_date:
            day_dir = self._build_day_dir(target_date)
            candidates = sorted(day_dir.glob("processed_reddit_*.csv"))
        else:
            candidates = sorted(self.by_day_root.glob("**/processed_reddit_*.csv"))

        if not candidates:
            return self.by_day_root / "missing_processed_reddit.csv"

        return candidates[-1]

    @staticmethod
    def _build_day_dir(run_date: str) -> Path:
        try:
            year, month, day = run_date.split("-")
        except ValueError:
            raise ValueError(f"run_date must be YYYY-MM-DD, got {run_date}")
        return PROCESSED_REDDIT_BY_DAY_DIR / year / month / day

    def _safe_eval_list(self, x):
        """convert stringified list to list safely."""
        if isinstance(x, list):
            return x
        try:
            return eval(x) if isinstance(x, str) else []
        except:
            return []

    def _safe_eval_dict(self, x):
        """convert stringified dict to dict safely."""
        if isinstance(x, dict):
            return x
        try:
            return eval(x) if isinstance(x, str) else {}
        except:
            return {}

    # ------------------------------------------------------------
    # trending logic
    # ------------------------------------------------------------
    def _expand_tickers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        convert wide-format rows:
            tickers = ["NVDA", "AMD"]
            ticker_scores = {"NVDA": 0.9, "AMD": 0.4}
        into long-format rows:
            ticker, confidence, engagement, date, subreddit
        """
        rows = []
        
        # iterate through each row of the dataframe.
        for _, row in df.iterrows():
            # get the tickers and scores.
            tickers = row.get("tickers", [])
            scores = row.get("ticker_scores", {})
            engagement = row.get("engagement", 0)
            date = row.get("date", None)
            subreddit = row.get("subreddit", None)
            context = row.get("cleaned_text", "")[:200]

            # iterate through each ticker.
            for ticker in tickers:
                conf = scores.get(ticker, 0)

                # append the row to the list.
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
        """aggregate and score each ticker."""

        grouped = df_long.groupby("ticker").agg({
            "confidence": "mean",
            "engagement": "sum",
            "ticker": "count",
        }).rename(columns={"ticker": "mentions"})

        # final score:
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
        """main entry: returns top trending tickers."""
        df = self._load_processed_df()
        if df.empty:
            return pd.DataFrame()

        # expand the tickers.
        df_long = self._expand_tickers(df)
        if df_long.empty:
            print(f"{Fore.YELLOW}no ticker data available in processed file.{Style.RESET_ALL}")
            return pd.DataFrame()

        # score the tickers.
        scoring = self._score_tickers(df_long)

        # save trending CSV.
        from src.utils.path_config import TICKER_GENERAL_DIR
        out_file = TICKER_GENERAL_DIR / f"trending_{df['date'].max()}.csv"
        scoring.to_csv(out_file, index=False)
        print(f"{Fore.GREEN}✓ saved trending ticker analysis to {out_file}{Style.RESET_ALL}")

        # return top N.
        return scoring.head(top_n)

    def get_trending_tickers(self, top_n: int = 10) -> List[str]:
        """return just the ticker symbols."""
        df = self.identify_trending(top_n=top_n)
        return df["ticker"].tolist() if not df.empty else []

