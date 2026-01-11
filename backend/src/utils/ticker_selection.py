# imports.
import pandas as pd
from typing import List
from colorama import Fore, Style
from __future__ import annotations

# local imports.
from src.utils.path_config import processed_reddit_by_day_dir
from src.utils.config import topN, min_mentions, min_engagement

def grab_top_tickers(processed: pd.DataFrame, run_id: str) -> List[str]:

    df = processed

    # -- 1. aggregate to a ticker-level table (across all days in this run).
    df["weighted_numer"] = df["post_sentiment"] * df["engagement"]

    # -- 2. group by ticker and aggregate.
    agg = df.groupby("ticker", as_index=False).agg(
        mention_count=("post_id", "nunique"),
        total_engagement=("engagement", "sum"),
        boost_score_sum=("boost_score", "sum"),
        weighted_numer=("weighted_numer", "sum"),
    )

    # -- 3. calculate weighted sentiment.
    agg["weighted_sentiment"] = agg.apply(
        lambda r: (r["weighted_numer"] / r["total_engagement"]) if r["total_engagement"] else 0.0,
        axis=1,
    )
    
    # -- 4. drop the weighted numer column.
    agg = agg.drop(columns=["weighted_numer"], errors="ignore")

    # -- 5. keep everything with at least 1 mention (no hard engagement gate yet).
    filtered = agg[agg["mention_count"] >= 1].copy()

    # -- 6. calculate trend strength and filter by engagement.
    filtered["trend_strength"] = filtered["boost_score_sum"] * (filtered["total_engagement"] + 1) ** 0.5
    filtered["meets_min_engagement"] = (filtered["total_engagement"] >= min_engagement).astype(int)
    filtered["rank_score"] = filtered["trend_strength"] + (filtered["meets_min_engagement"] * 0.01)
    top = filtered.sort_values("rank_score", ascending=False).head(topN)

    # -- 7. write watchlist csv for stage 3 handoff.
    processed_reddit_by_day_dir.mkdir(parents=True, exist_ok=True)
    watchlist_path = processed_reddit_by_day_dir / f"watchlist_{run_id}.csv"
    top.to_csv(watchlist_path, index=False)
    print(f"{Fore.CYAN}saved stage 3 watchlist to {Style.RESET_ALL}{watchlist_path.name}")

    # -- 8. return top tickers.
    return top["ticker"].dropna().astype(str).tolist()


