from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from colorama import Fore, Style

from src.utils.path_config import processed_reddit_by_day_dir, processed_metrics_dir, processed_stocktwits_by_day_dir


class FeatureBuilder:
    """
    Stage 5 (merge + features): join daily reddit metrics with daily OHLCV, then build
    a minimal feature set + next-day label for training.

    Sentiment source is ``reddit_daily_all.csv`` (master from reddit_processor). Tickers
    are restricted to those with ``raw_{TICKER}.csv`` under ``stocks_by_ticker_dir`` so
    each run rebuilds the full merge from accumulated Reddit + all stock files on disk.

    Output shape (one row per ticker-date):
      - keys: ticker, date
      - features: sentiment, buzz, chg, lags/rolls (milestone B), had_reddit, close_ret_3d
      - label: y_ret_1d
    """

    def build_dataset(
        self,
        stocks_by_ticker_dir: Path,
        run_id: str,
        tickers: Optional[List[str]] = None,
    ) -> Path:

        # create directory for merged features.
        processed_metrics_dir.mkdir(parents=True, exist_ok=True)

        reddit_daily_path = processed_reddit_by_day_dir / "reddit_daily_all.csv"
        if not reddit_daily_path.exists():
            out_path = processed_metrics_dir / f"merged_features_{run_id}.csv"
            pd.DataFrame([]).to_csv(out_path, index=False)
            print(f"{Fore.YELLOW}stage 5 - no {reddit_daily_path.name} yet; wrote empty {out_path.name}{Style.RESET_ALL}")
            return processed_metrics_dir / "merged_features_all.csv"

        daily = pd.read_csv(reddit_daily_path)
        if daily.empty:
            out_path = processed_metrics_dir / f"merged_features_{run_id}.csv"
            pd.DataFrame([]).to_csv(out_path, index=False)
            return processed_metrics_dir / "merged_features_all.csv"

        # normalize date key (master table uses "date"; legacy per-run files used "created_date").
        daily = daily.copy()
        daily["ticker"] = daily["ticker"].astype(str)
        if "date" in daily.columns:
            daily["date"] = daily["date"].astype(str)
        elif "created_date" in daily.columns:
            daily["date"] = daily["created_date"].astype(str)
            daily = daily.drop(columns=["created_date"], errors="ignore")
        else:
            print(f"{Fore.RED}stage 5 - reddit daily missing date/created_date column{Style.RESET_ALL}")
            out_path = processed_metrics_dir / f"merged_features_{run_id}.csv"
            pd.DataFrame([]).to_csv(out_path, index=False)
            return processed_metrics_dir / "merged_features_all.csv"

        stocks_dir = Path(stocks_by_ticker_dir)
        have_stocks = set()
        for p in stocks_dir.glob("raw_*.csv"):
            stem = p.stem
            have_stocks.add(stem[4:] if stem.startswith("raw_") else stem)
        daily = daily[daily["ticker"].isin(have_stocks)].copy()
        if daily.empty:
            out_path = processed_metrics_dir / f"merged_features_{run_id}.csv"
            pd.DataFrame([]).to_csv(out_path, index=False)
            print(f"{Fore.YELLOW}stage 5 - no overlap between reddit tickers and stock files under {stocks_dir}{Style.RESET_ALL}")
            return processed_metrics_dir / "merged_features_all.csv"

        # optional: restrict to an explicit ticker list (e.g. tonight's watchlist only).
        if tickers:
            want = {str(t) for t in tickers if t is not None}
            daily = daily[daily["ticker"].isin(want)].copy()

        # optional: merge stocktwits daily (if exists) onto the reddit daily table.
        st_path = processed_stocktwits_by_day_dir / f"stocktwits_ticker_daily_{run_id}.csv"
        if st_path.exists():
            st = pd.read_csv(st_path)
            if not st.empty:
                st["ticker"] = st["ticker"].astype(str)
                st["date"] = st["date"].astype(str)
                keep = ["ticker", "date", "st_mention_count", "st_total_likes", "st_weighted_sentiment"]
                st = st[[c for c in keep if c in st.columns]].copy()
                daily = daily.merge(st, on=["ticker", "date"], how="left")
                for c in ["st_mention_count", "st_total_likes", "st_weighted_sentiment"]:
                    if c in daily.columns:
                        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0)

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
            out_path = processed_metrics_dir / f"merged_features_{run_id}.csv"
            pd.DataFrame([]).to_csv(out_path, index=False)
            return processed_metrics_dir / "merged_features_all.csv"

        # concat the stock rows into a single dataframe.
        stocks = pd.concat(stock_rows, ignore_index=True)

        # merge the reddit daily metrics with the stock rows (inner join: only days where we have both sentiment + ohlcv).
        merged = daily.merge(stocks, on=["ticker", "date"], how="inner")

        # sort for sentiment-change feature.
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

        # --- had_reddit: 1 = reddit daily row has real mention activity for this ticker-day, 0 = missing or zero mentions.
        # inner join already guarantees a reddit row; we still set this from mention_count so later outer joins stay consistent.
        if "mention_count" in merged.columns:
            mc = pd.to_numeric(merged["mention_count"], errors="coerce").fillna(0)
            merged["had_reddit"] = (mc >= 1).astype(np.int8)
        else:
            merged["had_reddit"] = np.int8(1)

        n_no = int((merged["had_reddit"] == 0).sum())
        if n_no:
            print(
                f"{Fore.CYAN}stage 5 - had_reddit=0 (no mention_count): {n_no} rows{Style.RESET_ALL}"
            )

        # --- minimal feature set (computed from <= current day D, label uses D+1) ---
        # keep it small until we see backtest results.
        #   - weighted_sentiment: direction
        #   - buzz: log1p(total_engagement) (attention)
        #   - sentiment_chg_1d: sentiment acceleration
        #   - close_ret_3d: simple trend context
        #   - had_reddit: coverage flag (milestone A)

        for col in ["weighted_sentiment", "total_engagement", "close_ret_3d", "y_ret_1d"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        merged["buzz"] = np.log1p(merged.get("total_engagement", 0).fillna(0))
        merged["sentiment_chg_1d"] = merged.get("weighted_sentiment") - merged.groupby("ticker")["weighted_sentiment"].shift(1)

        # --- milestone B: per-ticker lags + rolling means (rows already sorted by ticker, date).
        # rolling window includes day D and earlier rows for that ticker only — no future dates.
        _tg = merged["ticker"]
        merged["weighted_sentiment_lag1"] = merged.groupby(_tg, sort=False)["weighted_sentiment"].shift(1)
        merged["buzz_lag1"] = merged.groupby(_tg, sort=False)["buzz"].shift(1)
        merged["weighted_sentiment_roll3_mean"] = merged.groupby(_tg, sort=False)["weighted_sentiment"].transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )
        merged["weighted_sentiment_roll5_mean"] = merged.groupby(_tg, sort=False)["weighted_sentiment"].transform(
            lambda s: s.rolling(5, min_periods=1).mean()
        )

        # --- milestone C': day-over-day change in buzz (log1p engagement), not redundant with sentiment_chg_1d.
        merged["buzz_dod"] = merged["buzz"] - merged["buzz_lag1"]

        # write the merged dataframe to a csv file.
        merged["date"] = merged["date"].dt.date.astype(str)
        out_cols = [
            "ticker",
            "date",
            "weighted_sentiment",
            "buzz",
            "sentiment_chg_1d",
            "weighted_sentiment_lag1",
            "buzz_lag1",
            "buzz_dod",
            "weighted_sentiment_roll3_mean",
            "weighted_sentiment_roll5_mean",
            "had_reddit",
            "close_ret_3d",
            "y_ret_1d",
        ]
        merged = merged[[c for c in out_cols if c in merged.columns]].copy()
        run_path = processed_metrics_dir / f"merged_features_{run_id}.csv"
        merged.to_csv(run_path, index=False)

        # full recompute from master reddit × all on-disk stock files (single source of truth).
        master_path = processed_metrics_dir / "merged_features_all.csv"
        combined = merged.copy()
        combined["ticker"] = combined["ticker"].astype(str)
        combined["date"] = combined["date"].astype(str)
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last").sort_values(["date", "ticker"])
        combined.to_csv(master_path, index=False)

        if missing:
            print(
                f"{Fore.YELLOW}stage 5 - missing stock files for {len(missing)} tickers (skipped):{Style.RESET_ALL} "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )

        print(
            f"{Fore.CYAN}stage 5 - saved merged dataset to: {Style.RESET_ALL}{run_path.name} "
            f"{Fore.CYAN}({len(combined):,} rows in {Path(master_path).name}){Style.RESET_ALL}"
        )
        return master_path


