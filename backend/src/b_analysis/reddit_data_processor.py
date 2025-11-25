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

import re
import pandas as pd
from pathlib import Path
from colorama import Fore, Style
from typing import Dict, List

from src.b_analysis.ticker_extractor import TickerExtractor
from src.b_analysis.entity_linker import EntityLinker  # minimal version
from src.utils.path_config import REDDIT_DATA_DIR, PROCESSED_REDDIT_DIR


class RedditDataProcessor:
    def __init__(self):
        print(f"{Fore.CYAN}Initializing Reddit Data Processor...{Style.RESET_ALL}")

        self.raw_dir = REDDIT_DATA_DIR
        self.output_file = PROCESSED_REDDIT_DIR / "processed_reddit.csv"

        # Modules
        self.extractor = TickerExtractor()
        self.entity_linker = EntityLinker()

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
        print(f"{Fore.GREEN}✓ Loaded {len(df)} raw posts{Style.RESET_ALL}")

        text_col = "text" if "text" in df.columns else "selftext"

        df["cleaned_text"] = df[text_col].fillna("") + " " + df["title"].fillna("")
        df["cleaned_text"] = df["cleaned_text"].apply(self._clean_text)

        # Relevance
        df["is_relevant"] = df["cleaned_text"].apply(self._is_financial_post)

        # Engagement
        df["engagement"] = df["score"].fillna(0) + df["num_comments"].fillna(0)

        # Extract tickers + confidence
        tickers_list = []
        scores_list = []

        for text in df["cleaned_text"]:
            result = self.extractor.extract_tickers(text)
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

        print(f"{Fore.GREEN}✓ Saved processed reddit data → {self.output_file}{Style.RESET_ALL}")
        return df


# ------------------------------------------------------------
# Direct test run
# ------------------------------------------------------------
if __name__ == "__main__":
    processor = RedditDataProcessor()
    processor.process()
