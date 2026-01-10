from __future__ import annotations

# part B of the collection phase.
#
# StockTwits is collected *after* Reddit processing, because we need the
# reddit-derived ticker watchlist first. So this module supports a single
# "collect + process" call for orchestrator simplicity.

import csv
import json
import requests
from pathlib import Path
from colorama import Fore, Style
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# local imports.
from src.b_analysis.sentiment_scorer import SentimentScorer
from src.utils.config import stocktwits_limit_per_ticker, stocktwits_score_sentiment
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

    def collect(self, tickers: List[str], stem: str, run_id: str) -> Path | None:

        # normalize tickers (dedupe, preserve order)
        tickers = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
        tickers = list(dict.fromkeys(tickers))
        if not tickers:
            return None
        
        # create the raw stocktwits directory.
        raw_stocktwits_dir.mkdir(parents=True, exist_ok=True)

        # NOTE: `stem` already includes the pipeline run id (because it comes from the reddit raw filename),
        # so we intentionally do NOT append `run_id` again.
        out_csv_path = raw_stocktwits_dir / f"stocktwits_messages_{stem}.csv"
        out_meta_path = raw_stocktwits_dir / f"stocktwits_meta_{stem}.json"
        out_fail_path = raw_stocktwits_dir / f"stocktwits_failures_{stem}.csv"

        # For downstream stages, keep the schema minimal + stable.
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

        wrote = 0
        failures: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {"run_id": run_id, "stem": stem, "tickers": {}, "schema": fieldnames}

        with out_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for t in tickers:
                try:
                    payload = self.fetch_symbol_stream(t, stocktwits_limit_per_ticker)
                except Exception as e:
                    failures.append({"ticker": t, "error": str(e)})
                    continue

                if not isinstance(payload, dict):
                    failures.append({"ticker": t, "error": "non-dict payload"})
                    continue

                meta["tickers"][t] = {
                    "symbol": payload.get("symbol"),
                    "cursor": payload.get("cursor"),
                }

                msgs = payload.get("messages") or []
                if not isinstance(msgs, list):
                    continue

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

        # Save meta so we can debug and paginate later without re-downloading everything.
        with out_meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if failures:
            with out_fail_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["ticker", "error"])
                w.writeheader()
                for item in failures:
                    w.writerow({"ticker": item.get("ticker", ""), "error": item.get("error", "")})

        print(
            f"{Fore.CYAN}stage 2.5 - saved stocktwits messages to {Style.RESET_ALL}{out_csv_path.name} "
            f"({wrote} msgs)"
        )
        return out_csv_path

    def collect_and_process(
        self,
        tickers: List[str],
        stem: str,
        run_id: str,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Stage 2B (combined): collect raw messages (2.5) then aggregate into daily metrics (2.6).
        Returns (messages_csv_path, daily_csv_path). Either can be None.
        """
        messages_path = self.collect(tickers=tickers, stem=stem, run_id=run_id)
        if not messages_path or not Path(messages_path).exists():
            return None, None

        try:
            print(f"{Fore.CYAN}=== stage 2.6 : stocktwits data processing ==={Style.RESET_ALL}")

            df = pd.read_csv(messages_path)
            if df.empty:
                return messages_path, None

            df = df.copy()
            df["ticker"] = df["ticker"].astype(str)
            df["created_at"] = pd.to_datetime(df.get("created_at"), utc=True, errors="coerce")
            df = df.dropna(subset=["ticker", "created_at"])
            df["date"] = df["created_at"].dt.date.astype(str)

            df["likes_total"] = pd.to_numeric(df.get("likes_total", 0), errors="coerce").fillna(0).astype(float)

            scorer = None
            if stocktwits_score_sentiment:
                try:
                    scorer = SentimentScorer()
                except Exception as e:
                    print(f"{Fore.YELLOW}stocktwits sentiment scoring disabled (init failed): {e}{Style.RESET_ALL}")

            if scorer is not None:
                scores: List[float] = []
                for body in df.get("body", "").fillna("").astype(str).tolist():
                    try:
                        scores.append(float(scorer.score(body, subreddit=None).get("score", 0.0)))
                    except Exception:
                        scores.append(0.0)
                df["sentiment_score"] = np.array(scores, dtype=float)
            else:
                df["sentiment_score"] = 0.0

            df["weight"] = 1.0 + np.log1p(df["likes_total"])
            df["weighted_numer"] = df["sentiment_score"] * df["weight"]

            daily = (
                df.groupby(["date", "ticker"], as_index=False)
                .agg(
                    st_mention_count=("ticker", "size"),
                    st_total_likes=("likes_total", "sum"),
                    weighted_numer=("weighted_numer", "sum"),
                    weight_sum=("weight", "sum"),
                )
            )
            daily["st_weighted_sentiment"] = daily.apply(
                lambda r: (r["weighted_numer"] / r["weight_sum"]) if r["weight_sum"] else 0.0,
                axis=1,
            )
            daily = daily.drop(columns=["weighted_numer", "weight_sum"])

            processed_stocktwits_by_day_dir.mkdir(parents=True, exist_ok=True)
            daily_path = processed_stocktwits_by_day_dir / f"stocktwits_ticker_daily_{stem}.csv"
            daily.to_csv(daily_path, index=False)

            print(f"saved stocktwits ticker-daily output to {Fore.YELLOW}{daily_path.name}{Style.RESET_ALL}")
            return messages_path, daily_path
        except Exception as e:
            print(f"{Fore.YELLOW}stage 2.6 - stocktwits processing skipped: {e}{Style.RESET_ALL}")
            return messages_path, None


