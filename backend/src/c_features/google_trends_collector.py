from __future__ import annotations

import time
import random
import logging
import warnings
from colorama import Fore, Style
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from pytrends.request import TrendReq

from src.utils.path_config import processed_trends_by_day_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrendsResult:
    ok: bool
    path: Optional[str]
    rows: int = 0
    error: Optional[str] = None


class GoogleTrendsCollector:
    """
    Lightweight "attention proxy" source:
      - For each ticker, query Google Trends interest over time for "<TICKER> stock"
      - Save daily rows: date, ticker, trends_interest (+ change features)

    Notes:
      - Google Trends is not a formal public API; pytrends can be rate-limited/blocked.
      - Keep this non-fatal in the pipeline.
    """

    def __init__(self, geo: str = "US", timeframe: str = "now 7-d", max_retries: int = 5, base_delay_s: float = 3.0, verbose: bool = False,):
        
        self.geo = geo
        self.timeframe = timeframe
        self.max_retries = int(max_retries)
        self.base_delay_s = float(base_delay_s)
        self.verbose = bool(verbose)

        # pytrends currently emits a noisy pandas FutureWarning inside its own internals.
        # Keep runs clean by ignoring that specific warning source.
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"pytrends\.request")

        # tz=0 => UTC-ish bucket; hl en-US.
        # Add a browser-like UA. Trends is not an official API; this reduces random bot-blocks a bit.
        self.pytrends = TrendReq(
            hl="en-US",
            tz=0,
            requests_args={
                "headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                }
            },
        )

    def _sleep_backoff(self, attempt: int) -> float:
        # exponential backoff + jitter
        delay = self.base_delay_s * (2 ** max(0, attempt))
        delay += random.uniform(0.0, 2.0)
        time.sleep(delay)
        return delay

    def _call_with_retry(self, fn, *args, op: str, batch_info: str, **kwargs):
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self.verbose:
                    logger.warning(f"[trends] {op} attempt {attempt+1}/{self.max_retries} {batch_info}")
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                slept = self._sleep_backoff(attempt)
                if self.verbose:
                    logger.warning(
                        f"[trends] {op} failed (attempt {attempt+1}/{self.max_retries}) {batch_info} "
                        f"sleep={slept:.1f}s err={e}"
                    )
        if last_err:
            raise last_err
        return None

    @staticmethod
    def build_terms(tickers: List[str]) -> List[Tuple[str, str]]:
        # -- 1. return [(ticker, query_term)]
        out: List[Tuple[str, str]] = []
        # -- 2. iterate through the tickers and build the terms.
        for t in tickers or []:
            # -- 2.1 check if the ticker is valid.
            if not t:
                continue
            # -- 2.2 convert the ticker to uppercase.
            sym = str(t).strip().upper()
            if not sym:
                continue
            # -- 2.3 add the ticker and query term to the output.
            out.append((sym, f"{sym} stock"))
        return out

    def collect(self, tickers: List[str], run_id: str) -> TrendsResult:
        try:
            print(f"{Fore.CYAN}=== stage 3 : google trends collection ==={Style.RESET_ALL}")

            # -- 1. create the output directory.
            processed_trends_by_day_dir.mkdir(parents=True, exist_ok=True)
            out_path = processed_trends_by_day_dir / f"google_trends_daily_{run_id}.csv"
            master_path = processed_trends_by_day_dir / "google_trends_daily_all.csv"

            # -- 2. build the terms.
            pairs = self.build_terms(tickers)
            if not pairs:
                pd.DataFrame([]).to_csv(out_path, index=False)
                return TrendsResult(ok=False, path=str(out_path), rows=0, error="no tickers to query")

            # -- 3. build the payload.
            rows = []
            batch_size = 5

            # -- 4. iterate through the pairs and build the payload.
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                kw_list = [term for _, term in batch]
                batch_info = f"(run_id={run_id} geo={self.geo} timeframe={self.timeframe} batch={i//batch_size+1} kws={kw_list})"

                # Small delay to reduce rate-limiting.
                if i:
                    time.sleep(1.0 + random.uniform(0.0, 1.0))

                try:
                    self._call_with_retry(
                        self.pytrends.build_payload,
                        op="build_payload",
                        batch_info=batch_info,
                        kw_list=kw_list,
                        timeframe=self.timeframe,
                        geo=self.geo,
                    )
                    df = self._call_with_retry(
                        self.pytrends.interest_over_time,
                        op="interest_over_time",
                        batch_info=batch_info,
                    )
                except Exception:
                    # skip this batch (non-fatal)
                    if self.verbose:
                        logger.warning(f"[trends] skipping batch after retries {batch_info}")
                    continue

                if df is None or df.empty:
                    continue

                # index is datetime; columns are kw terms + "isPartial"
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])

                df = df.reset_index().rename(columns={"date": "date"})

                for ticker, term in batch:
                    if term not in df.columns:
                        continue
                    sub = df[["date", term]].copy()
                    sub = sub.rename(columns={term: "trends_interest"})
                    sub["ticker"] = ticker
                    rows.append(sub)

            if not rows:
                pd.DataFrame([]).to_csv(out_path, index=False)
                return TrendsResult(ok=False, path=str(out_path), rows=0, error="no trends rows returned (rate-limited or empty data)")

            combined = pd.concat(rows, ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date.astype(str)
            combined["trends_interest"] = pd.to_numeric(combined["trends_interest"], errors="coerce").fillna(0).astype(float)
            combined = combined.dropna(subset=["date", "ticker"])

            # ---- collapse to ONE row per (date, ticker) ----
            # pytrends can return hourly granularity; for a daily pipeline, aggregate.
            daily = (
                combined.groupby(["date", "ticker"], as_index=False)
                .agg(trends_interest=("trends_interest", "max"))
                .sort_values(["ticker", "date"])
                .reset_index(drop=True)
            )

            # ---- day-over-day + rolling baseline features (per ticker) ----
            # these are more useful than raw levels because Trends is scaled/relative.
            daily["trends_dod"] = daily.groupby("ticker")["trends_interest"].diff(1)

            prev = daily.groupby("ticker")["trends_interest"].shift(1)
            daily["trends_dod_pct"] = (daily["trends_interest"] / prev.replace({0.0: pd.NA})) - 1.0
            daily["trends_dod_pct"] = pd.to_numeric(daily["trends_dod_pct"], errors="coerce").fillna(0.0)

            roll = daily.groupby("ticker")["trends_interest"].rolling(window=7, min_periods=2)
            daily["trends_roll7_mean"] = roll.mean().reset_index(level=0, drop=True)
            daily["trends_roll7_std"] = roll.std(ddof=0).reset_index(level=0, drop=True)

            # z-score vs rolling baseline (guard std=0)
            std = daily["trends_roll7_std"].replace({0.0: pd.NA})
            daily["trends_roll7_z"] = (daily["trends_interest"] - daily["trends_roll7_mean"]) / std
            daily["trends_roll7_z"] = pd.to_numeric(daily["trends_roll7_z"], errors="coerce").fillna(0.0)

            # ratio vs rolling mean (guard mean=0)
            mean = daily["trends_roll7_mean"].replace({0.0: pd.NA})
            daily["trends_roll7_ratio"] = (daily["trends_interest"] / mean) - 1.0
            daily["trends_roll7_ratio"] = pd.to_numeric(daily["trends_roll7_ratio"], errors="coerce").fillna(0.0)

            # save run-specific file
            daily.to_csv(out_path, index=False)

            # ---- historical feature table (append + dedupe by date,ticker) ----
            try:
                if master_path.exists():
                    old = pd.read_csv(master_path)
                    combined_all = pd.concat([old, daily], ignore_index=True)
                else:
                    combined_all = daily.copy()

                combined_all["date"] = combined_all["date"].astype(str)
                combined_all["ticker"] = combined_all["ticker"].astype(str)

                # keep latest value for the same (date,ticker) if duplicated
                combined_all = (
                    combined_all.drop_duplicates(subset=["date", "ticker"], keep="last")
                    .sort_values(["date", "ticker"])
                    .reset_index(drop=True)
                )
                combined_all.to_csv(master_path, index=False)
            except Exception as e:
                if self.verbose:
                    logger.warning(f"[trends] failed to update master table: {e}")

            return TrendsResult(ok=True, path=str(out_path), rows=int(len(daily)))
        except Exception as e:
            return TrendsResult(ok=False, path=None, rows=0, error=str(e))

