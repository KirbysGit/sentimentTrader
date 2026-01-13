from __future__ import annotations

# FUCK!!!!
# stocktwits doesn't allow for requests to be made without a user-agent.
# so this is como se dice. "chopped".

import csv
import json
import requests
from pathlib import Path
from colorama import Fore, Style
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# local imports.
from src.b_analysis.helpers.sentiment_scorer import SentimentScorer
from src.utils.config import stocktwits_limit_per_ticker
from src.utils.path_config import raw_stocktwits_dir
from src.utils.path_config import processed_stocktwits_by_day_dir


class StocktwitsCollector:

    # set base url for fetching.
    BASE = "https://api.stocktwits.com/api/2"

    # --- helper functions.

    # fetch full stream payload for a given ticker (messages + cursor + symbol).
    def fetch_symbol_stream(self, ticker: str, limit: int) -> Dict[str, Any]:
        # -- 1. build url.
        url = f"{self.BASE}/streams/symbol/{ticker}.json"

        # -- 2. fetch messages.
        r = requests.get(url, params={"limit": int(limit)}, timeout=20)
        r.raise_for_status()

        # -- 3. return payload.
        return r.json() or {}
    
    @staticmethod
    # safer integer extraction from nested dictionary.
    def get_int(d: Any, *path: str, default: int = 0) -> int:
        cur: Any = d
        for k in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
        try:
            return int(cur)
        except Exception:
            return default

    @staticmethod
    # safer string extraction from nested dictionary.
    def get_str(d: Any, *path: str, default: str = "") -> str:
        cur: Any = d
        for k in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
        return default if cur is None else str(cur)

    @staticmethod
    # get mentioned symbols from a message.
    def get_mentioned_symbols(msg: Dict[str, Any]) -> str:
        # -- 1. get symbols from message.
        syms = msg.get("symbols") or []

        # -- 2. if not a list, return empty string.
        if not isinstance(syms, list):
            return ""

        # -- 3. initialize our output list.
        out: List[str] = []

        # -- 4. iterate through symbols.
        for s in syms:
            if isinstance(s, dict) and s.get("symbol"):
                out.append(str(s["symbol"]).upper())

        # -- 5. de-dupe but keep stable readable output.
        return ",".join(sorted(set(out)))
    

    # --- main collection function.

    def collect(self, tickers: List[str], run_id: str) -> Path | None:

        # -- 1. create the raw stocktwits directory.
        raw_stocktwits_dir.mkdir(parents=True, exist_ok=True)

        # -- 2. create the output paths.
        out_csv_path = raw_stocktwits_dir / f"stocktwits_messages_{run_id}.csv"
        out_meta_path = raw_stocktwits_dir / f"stocktwits_meta_{run_id}.json"
        out_fail_path = raw_stocktwits_dir / f"stocktwits_failures_{run_id}.csv"

        # -- 3. define the fieldnames for the output csv.
        fieldnames = [
            "ticker",
            "message_id",
            "created_at",
            "body",
            "likes_total",
            "user_followers",
            "st_basic_sentiment",
            "mentioned_symbols",
        ]

        # -- 4. initialize counters and metadata.
        wrote = 0
        failures: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {"run_id": run_id, "tickers": {}, "schema": fieldnames}

        # -- 5. open the output csv and write the header.
        with out_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # -- 6. iterate through the tickers and fetch the symbol stream.
            for t in tickers:
                try:
                    # -- 6.1 grab json on ticker page.
                    payload = self.fetch_symbol_stream(t, stocktwits_limit_per_ticker)
                except Exception as e:
                    failures.append({"ticker": t, "error": str(e)})
                    continue

                # -- 6.2 check if payload is a dictionary.
                if not isinstance(payload, dict):
                    failures.append({"ticker": t, "error": "non-dict payload"})
                    continue

                # -- 6.3 add metadata to meta dictionary.
                meta["tickers"][t] = {
                    "symbol": payload.get("symbol"),
                    "cursor": payload.get("cursor"),
                }

                # -- 6.4 get messages from payload.
                msgs = payload.get("messages") or []
                if not isinstance(msgs, list):
                    continue

                # -- 6.5 iterate through messages and write to csv.
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    row = {
                        "ticker": t,
                        "message_id": self.get_int(m, "id", default=0),
                        "created_at": self.get_str(m, "created_at"),
                        "body": self.get_str(m, "body"),
                        "likes_total": self.get_int(m, "likes", "total", default=0),
                        "user_followers": self.get_int(m, "user", "followers", default=0),
                        "st_basic_sentiment": self.get_str(m, "entities", "sentiment", "basic", default=""),
                        "mentioned_symbols": self.get_mentioned_symbols(m),
                    }
                    writer.writerow(row)
                    wrote += 1

        # save meta for debugging and pagination.
        with out_meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # save failures for debugging.
        if failures:
            with out_fail_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["ticker", "error"])
                w.writeheader()
                for item in failures:
                    w.writerow({"ticker": item.get("ticker", ""), "error": item.get("error", "")})

        # print success message.
        print(
            f"{Fore.CYAN}stage 3 - saved stocktwits messages to {Style.RESET_ALL}{out_csv_path.name} "
            f"({wrote} msgs)"
        )

        # return the path to the messages csv.
        return out_csv_path

    def collect_and_process(self, tickers: List[str], run_id: str) -> Tuple[Optional[Path], Optional[Path]]:
        
        # -- 1. collect the messages.
        messages_path = self.collect(tickers=tickers, run_id=run_id)

        # -- 2. check if the messages path exists.
        if not messages_path or not Path(messages_path).exists():
            return None, None

        try:
            print(f"{Fore.CYAN}=== stage 3.5 - stocktwits data processing ==={Style.RESET_ALL}")

            # -- 2.1 read the messages csv.
            df = pd.read_csv(messages_path)
            if df.empty:
                return messages_path, None

            # -- 2.2 convert the dataframe to the correct type.
            df = df.copy()
            df["ticker"] = df["ticker"].astype(str)
            df["created_at"] = pd.to_datetime(df.get("created_at"), utc=True, errors="coerce")
            df = df.dropna(subset=["ticker", "created_at"])
            df["date"] = df["created_at"].dt.date.astype(str)

            df["likes_total"] = pd.to_numeric(df.get("likes_total", 0), errors="coerce").fillna(0).astype(float)

            # -- 2.3 initialize the sentiment scorer.
            scorer = SentimentScorer()

            # -- 2.4 score the sentiment of the body.
            scores: List[float] = []
            for body in df.get("body", "").fillna("").astype(str).tolist():
                try:
                    scores.append(float(scorer.score(body, subreddit=None).get("score", 0.0)))
                except Exception:
                    scores.append(0.0)

            # -- 2.5 add the sentiment scores to the dataframe.
            df["sentiment_score"] = np.array(scores, dtype=float)

            # -- 2.6 calculate the weight.
            df["weight"] = 1.0 + np.log1p(df["likes_total"])

            # -- 2.7 calculate the weighted numerator.
            df["weighted_numer"] = df["sentiment_score"] * df["weight"]

            # -- 2.8 group by date and ticker and aggregate.
            daily = (
                df.groupby(["date", "ticker"], as_index=False)
                .agg(
                    st_mention_count=("ticker", "size"),
                    st_total_likes=("likes_total", "sum"),
                    weighted_numer=("weighted_numer", "sum"),
                    weight_sum=("weight", "sum"),
                )
            )

            # -- 2.9 calculate the weighted sentiment.
            daily["st_weighted_sentiment"] = daily.apply(
                lambda r: (r["weighted_numer"] / r["weight_sum"]) if r["weight_sum"] else 0.0,
                axis=1,
            )

            # -- 2.10 drop the weighted numer and weight sum columns.
            daily = daily.drop(columns=["weighted_numer", "weight_sum"])

            # -- 2.11 create the output directory and save the dataframe.
            processed_stocktwits_by_day_dir.mkdir(parents=True, exist_ok=True)
            daily_path = processed_stocktwits_by_day_dir / f"stocktwits_ticker_daily_{stem}.csv"
            daily.to_csv(daily_path, index=False)

            print(f"saved stocktwits ticker-daily output to {Fore.YELLOW}{daily_path.name}{Style.RESET_ALL}")
            return messages_path, daily_path
        except Exception as e:
            print(f"{Fore.YELLOW}stage 2.6 - stocktwits processing skipped: {e}{Style.RESET_ALL}")
            return messages_path, None


