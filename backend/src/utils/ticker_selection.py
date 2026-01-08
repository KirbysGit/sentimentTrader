from __future__ import annotations
import pandas as pd
from typing import List
from pathlib import Path

from colorama import Fore, Style

from src.utils.path_config import processed_reddit_by_day_dir
from src.utils.config import topN, min_mentions, min_engagement


def grab_top_tickers(raw_output_path: Path) -> List[str]:

    # -- 1. get filename from raw output path.
    stem = Path(raw_output_path).stem
    
    # -- 2. get daily ticker metrics.
    daily_path = processed_reddit_by_day_dir / f"reddit_ticker_daily_{stem}.csv"
    daily = pd.read_csv(daily_path)

    # -- 3. keep everything with at least 1 mention (no hard engagement gate).
    # trend_strength below already rewards engagement; this prevents quiet days from collapsing to 0-1 tickers.
    filtered = daily[daily["mention_count"] >= 1].copy()

    # -- 4. aggregate by ticker so we get a unique top-N list (no duplicates).
    agg = filtered.groupby("ticker", as_index=False).agg(
        mention_count=("mention_count", "sum"),
        total_engagement=("total_engagement", "sum"),
        boost_score_sum=("boost_score_sum", "sum"),
        weighted_sentiment=("weighted_sentiment", "mean"),
        subreddit_diversity=("subreddit_diversity", "max"),
    )

    # -- 5. calculate trend strength and get top N.
    agg["trend_strength"] = agg["boost_score_sum"] * (agg["total_engagement"] + 1) ** 0.5
    agg["meets_min_engagement"] = (agg["total_engagement"] >= min_engagement).astype(int)
    agg["rank_score"] = agg["trend_strength"] + (agg["meets_min_engagement"] * 0.01)
    top = agg.sort_values("rank_score", ascending=False).head(topN)

    # -- 6. write watchlist csv for stage 3 handoff.
    watchlist_path = processed_reddit_by_day_dir / f"reddit_stage3_watchlist_{stem}.csv"
    top.to_csv(watchlist_path, index=False)
    print(f"{Fore.CYAN}saved stage 3 watchlist to {Style.RESET_ALL}{watchlist_path.name}")

    # -- 7. return top tickers.
    return top["ticker"].dropna().astype(str).tolist()


