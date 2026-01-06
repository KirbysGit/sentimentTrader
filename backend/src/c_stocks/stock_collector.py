import pandas as pd
import yfinance as yf
from typing import List
from datetime import datetime, timezone

# local imports.
from src.utils.path_config import raw_stocks_dir
from src.utils.config import lookback_days, topN

class StockCollector:

    def collect_stock_data(self, tickers: List[str]):
        # -- 1. creates raw stock directory to hold stock data.
        raw_stocks_dir.mkdir(parents=True, exist_ok=True)

        # -- 2. normalize tickers and write "ready-to-fetch" list.
        tickers = tickers[:topN]

        # -- 3. fetch daily OHLCV (simple yfinance v0).

        rows = []
        for t in tickers:
            df = yf.download(
                t,
                period=f"{lookback_days}d",
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
            if df is None or df.empty:
                continue

            df = df.reset_index()
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            if "date" not in df.columns:
                if "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "date"})
                elif "index" in df.columns:
                    df = df.rename(columns={"index": "date"})

            df["ticker"] = t
            rows.append(df)

        out_path = raw_stocks_dir / f"stocks_raw_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        if rows:
            out = pd.concat(rows, ignore_index=True)
            out.to_csv(out_path, index=False)
        else:
            pd.DataFrame([]).to_csv(out_path, index=False)

        return out_path


