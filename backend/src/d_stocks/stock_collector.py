from __future__ import annotations

import pandas as pd
import yfinance as yf
from typing import List
from pathlib import Path
from datetime import datetime, timezone
from colorama import Fore, Style

# local imports.
from src.utils.path_config import raw_stocks_dir, processed_stocks_by_day_dir
from src.utils.config import lookback_days, topN

class StockCollector:

    def collect_stock_data(self, tickers: List[str], run_id: str | None = None) -> Path:
        
        print(f"{Fore.CYAN}=== stage 4 : stock data collection ==={Style.RESET_ALL}")

        # -- 1. creates raw stock directory to hold stock data.
        raw_stocks_dir.mkdir(parents=True, exist_ok=True)
        by_ticker_dir = raw_stocks_dir / "by_ticker"
        by_ticker_dir.mkdir(parents=True, exist_ok=True)

        # -- 2. normalize tickers and write "ready-to-fetch" list.
        tickers = tickers[:topN]

        # -- 3. fetch daily OHLCV (simple yfinance v0).
        rows = []
        failures = []
        for t in tickers:

            # get the dataframe per ticker.
            try:
                # if we already have a file for this ticker, only fetch a small recent window and dedupe by date.

                ticker_path = by_ticker_dir / f"raw_{t}.csv"
                period_days = 7 if ticker_path.exists() else lookback_days

                df = yf.download(
                    t,
                    period=f"{period_days}d",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                )
            except Exception as e:
                failures.append({"ticker": t, "reason": str(e)})
                continue

            if df is None or df.empty:
                # yahoo/yfinance sometimes returns empty frames (delisted, bad symbol, etc.)
                failures.append({"ticker": t, "reason": "empty_or_none"})
                continue

            # yfinance sometimes returns MultiIndex columns (price_field, ticker). If so, keep only this ticker's slice.
            if isinstance(df.columns, pd.MultiIndex):
                lvl1 = [str(x).upper() for x in df.columns.get_level_values(1)]
                if t in lvl1:
                    mask = [str(x).upper() == t for x in df.columns.get_level_values(1)]
                    df = df.loc[:, mask]
                    df.columns = df.columns.get_level_values(0)
                else:
                    # fallback: drop to first level (best-effort).
                    df.columns = df.columns.get_level_values(0)
                
            # yfinance usually returns a DatetimeIndex; make it a real column for CSV + grouping.
            df = df.reset_index()

            # normalize column names to a simple schema.
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

            # ensure we always have a "date" column name (depends on our yfinance/pandas version).
            if "date" not in df.columns:
                if "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "date"})
                elif "index" in df.columns:
                    df = df.rename(columns={"index": "date"})

            # normalize date to a simple YYYY-MM-DD string.
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
                df = df.dropna(subset=["date"])

            # tag rows so we can concat multiple tickers into one table.
            df["ticker"] = t

            # keep only the standard columns (long format), with ticker first.
            keep = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
            df = df[[c for c in keep if c in df.columns]]
            rows.append(df)

            # write/update per-ticker file (dedupes by date).
            if ticker_path.exists():
                existing = pd.read_csv(ticker_path)
                if "date" in existing.columns:
                    existing["date"] = existing["date"].astype(str)
                merged = pd.concat([existing, df], ignore_index=True)
                if "date" in merged.columns:
                    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
                merged.to_csv(ticker_path, index=False)
            else:
                df.to_csv(ticker_path, index=False)

        # run id for output naming (prefer pipeline run id if provided)
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # write per-ticker failures (so we can see what got skipped).
        if failures:
            fail_path = raw_stocks_dir / f"stocks_failed_{run_id}.csv"
            pd.DataFrame(failures).to_csv(fail_path, index=False)

        # -- build per-day, per-ticker labels/features for training (master table)
        # -- converts per-ticker raw OHLCV into a single table keyed by (date, ticker).
        try:
            processed_stocks_by_day_dir.mkdir(parents=True, exist_ok=True)

            # -- concat the rows into a single dataframe.
            if rows:
                stocks = pd.concat(rows, ignore_index=True)
                stocks["ticker"] = stocks["ticker"].astype(str)
                stocks["date"] = stocks["date"].astype(str)
                stocks["close"] = pd.to_numeric(stocks.get("close"), errors="coerce")
                stocks = stocks.dropna(subset=["ticker", "date", "close"]).sort_values(["ticker", "date"])

                # -- calculate the labels/features (no lookahead leakage except for the label itself)
                stocks["close_ret_3d"] = stocks.groupby("ticker")["close"].pct_change(3)
                stocks["next_close"] = stocks.groupby("ticker")["close"].shift(-1)
                stocks["y_ret_1d"] = (stocks["next_close"] / stocks["close"]) - 1.0

                # -- create the labels dataframe.
                labels = stocks[["ticker", "date", "close", "next_close", "y_ret_1d", "close_ret_3d"]].copy()
                labels_path = processed_stocks_by_day_dir / f"stock_labels_{run_id}.csv"
                labels.to_csv(labels_path, index=False)

                # -- update the master table: append + dedupe by date,ticker
                master_path = processed_stocks_by_day_dir / "stock_labels_all.csv"
                if master_path.exists():
                    old = pd.read_csv(master_path)
                    combined = pd.concat([old, labels], ignore_index=True)
                else:
                    combined = labels

                # -- convert the ticker and date columns to strings.
                combined["ticker"] = combined["ticker"].astype(str)
                combined["date"] = combined["date"].astype(str)

                # -- drop duplicates and sort the dataframe.
                combined = (
                    combined.drop_duplicates(subset=["ticker", "date"], keep="last")
                    .sort_values(["date", "ticker"])
                    .reset_index(drop=True)
                )
                combined.to_csv(master_path, index=False)
        except Exception:
            # non-fatal: stock labels are nice-to-have; don't break collection
            pass

        # return where the per-ticker files live.
        return by_ticker_dir


