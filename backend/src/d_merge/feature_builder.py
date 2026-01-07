from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from colorama import Fore, Style

from src.utils.path_config import processed_reddit_by_day_dir, processed_metrics_dir


class FeatureBuilder:
    """
    Stage 4 (merge + features): join daily reddit metrics with daily OHLCV, then build
    a minimal feature set + next-day label for training.

    Output shape (one row per ticker-date):
      - keys: ticker, date
      - reddit daily metrics columns (as-is)
      - ohlcv columns (open/high/low/close/adj_close/volume)
      - features: minimal sentiment + price context (no lookahead)
      - labels: next_close, y_ret_1d
    """

    def build_dataset(self, raw_output_path: Path, stocks_by_ticker_dir: Path, tickers: Optional[List[str]] = None) -> Path:

        # create directory for merged features.
        processed_metrics_dir.mkdir(parents=True, exist_ok=True)

        # load reddit daily metrics.
        stem = Path(raw_output_path).stem
        reddit_daily_path = processed_reddit_by_day_dir / f"reddit_ticker_daily_{stem}.csv"

        # check if reddit daily metrics is empty.
        daily = pd.read_csv(reddit_daily_path)
        if daily.empty:
            out_path = processed_metrics_dir / f"merged_features_{stem}.csv"
            pd.DataFrame([]).to_csv(out_path, index=False)
            return out_path

        # unify date key.
        daily = daily.copy()
        daily["ticker"] = daily["ticker"].astype(str)
        daily["date"] = daily["created_date"].astype(str)
        daily = daily.drop(columns=[c for c in ["created_date"] if c in daily.columns])

        # build from specific tickers (stage 3 watchlist).
        if tickers:
            want = {str(t) for t in tickers if t is not None}
            daily = daily[daily["ticker"].isin(want)].copy()

        # load stock rows for tickers (long format).
        stock_rows = []
        missing = []

        # load stock rows for each ticker.
        for t in sorted(daily["ticker"].dropna().unique().tolist()):
            # get the path to the per-ticker stock file.
            p = Path(stocks_by_ticker_dir) / f"raw_{t}.csv"

            # check if the stock file exists.
            if not p.exists():
                missing.append(t)
                continue

            # load the stock file.
            s = pd.read_csv(p)
            if s.empty:
                missing.append(t)
                continue

            # normalize the stock file.
            s = s.copy()
            s["ticker"] = s["ticker"].astype(str)
            s["date"] = pd.to_datetime(s["date"], errors="coerce")
            s["close"] = pd.to_numeric(s.get("close"), errors="coerce")
            s = s.dropna(subset=["date", "close"])
            if s.empty:
                missing.append(t)
                continue

            # build price features + labels on the *full trading-day series*,
            # then merge onto sentiment days by (ticker, date).
            s = s.sort_values(["ticker", "date"]).reset_index(drop=True)
            s["close_ret_3d"] = s.groupby("ticker")["close"].pct_change(3)
            s["next_close"] = s.groupby("ticker")["close"].shift(-1)
            s["y_ret_1d"] = (s["next_close"] / s["close"]) - 1.0

            s["date"] = s["date"].dt.date.astype(str)
            s = s[["ticker", "date", "close_ret_3d", "y_ret_1d"]].copy()

            # add the stock row to the list.
            stock_rows.append(s)

        # check if there are any stock rows.
        if not stock_rows:
            # create an empty dataframe and save it.
            out_path = processed_metrics_dir / f"merged_features_{stem}.csv"
            pd.DataFrame([]).to_csv(out_path, index=False)
            return out_path

        # concat the stock rows into a single dataframe.
        stocks = pd.concat(stock_rows, ignore_index=True)

        # merge the reddit daily metrics with the stock rows (inner join: only days where we have both sentiment + ohlcv).
        merged = daily.merge(stocks, on=["ticker", "date"], how="inner")

        # sort for sentiment-change feature.
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

        # --- minimal feature set (computed from <= current day D, label uses D+1) ---
        # keep it small until we see backtest results.
        #   - weighted_sentiment: direction
        #   - buzz: log1p(total_engagement) (attention)
        #   - sentiment_chg_1d: sentiment acceleration
        #   - close_ret_3d: simple trend context 

        for col in ["weighted_sentiment", "total_engagement", "close_ret_3d", "y_ret_1d"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        merged["buzz"] = np.log1p(merged.get("total_engagement", 0).fillna(0))
        merged["sentiment_chg_1d"] = merged.get("weighted_sentiment") - merged.groupby("ticker")["weighted_sentiment"].shift(1)

        # write the merged dataframe to a csv file.
        merged["date"] = merged["date"].dt.date.astype(str)
        out_cols = ["ticker", "date", "weighted_sentiment", "buzz", "sentiment_chg_1d", "close_ret_3d", "y_ret_1d"]
        merged = merged[[c for c in out_cols if c in merged.columns]].copy()
        out_path = processed_metrics_dir / f"merged_features_{stem}.csv"
        merged.to_csv(out_path, index=False)

        # --- stack/aggregate across runs (training uses all history) ---
        master_path = processed_metrics_dir / "merged_features_all.csv"
        if master_path.exists():
            old = pd.read_csv(master_path)
            combined = pd.concat([old, merged], ignore_index=True)
        else:
            combined = merged.copy()

        # dedupe by ticker+date (keep latest run's row if duplicated).
        combined["ticker"] = combined["ticker"].astype(str)
        combined["date"] = combined["date"].astype(str)
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last").sort_values(["date", "ticker"])
        combined.to_csv(master_path, index=False)

        # print a warning if there are any missing stock files.
        if missing:
            print(
                f"{Fore.YELLOW}stage 4 - missing stock files for {len(missing)} tickers (skipped):{Style.RESET_ALL} "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )

        print(
            f"{Fore.CYAN}stage 4 - saved merged dataset to: {Style.RESET_ALL}{out_path.name} "
            f"{Fore.CYAN}(stacked to {Style.RESET_ALL}{master_path.name}{Fore.CYAN}){Style.RESET_ALL}"
        )
        return master_path


