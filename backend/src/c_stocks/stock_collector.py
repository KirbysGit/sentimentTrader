from __future__ import annotations

import pandas as pd
import yfinance as yf
from typing import List
from pathlib import Path
from datetime import datetime, timezone

# local imports.
from src.utils.path_config import raw_stocks_dir
from src.utils.config import lookback_days, topN

class StockCollector:

    def collect_stock_data(self, tickers: List[str]) -> Path:
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

            # tag rows so we can concat multiple tickers into one table.
            df["ticker"] = t

            # keep only the standard columns (long format), with ticker first.
            keep = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
            df = df[[c for c in keep if c in df.columns]]
            rows.append(df)

            # write/update per-ticker file (dedupes by date).
            if ticker_path.exists():
                existing = pd.read_csv(ticker_path)
                merged = pd.concat([existing, df], ignore_index=True)
                if "date" in merged.columns:
                    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
                merged.to_csv(ticker_path, index=False)
            else:
                df.to_csv(ticker_path, index=False)

        # only write per-ticker files (no combined "stocks_raw_*.csv").
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # write per-ticker failures (so we can see what got skipped).
        if failures:
            fail_path = raw_stocks_dir / f"stocks_failed_{run_id}.csv"
            pd.DataFrame(failures).to_csv(fail_path, index=False)

        # return where the per-ticker files live.
        return by_ticker_dir


