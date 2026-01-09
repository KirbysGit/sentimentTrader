from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import requests
from colorama import Fore, Style

from src.utils.config import stocktwits_limit_per_ticker
from src.utils.path_config import raw_stocktwits_dir


class StocktwitsCollector:

    # set base url for fetching.
    BASE = "https://api.stocktwits.com/api/2"

    # fetch messages for a given ticker.
    def fetch_symbol_messages(self, ticker: str, limit: int) -> List[Dict[str, Any]]:
        # build the url & fetch messages, handling errors if any.
        url = f"{self.BASE}/streams/symbol/{ticker}.json"
        r = requests.get(url, params={"limit": int(limit)}, timeout=20)
        r.raise_for_status()

        # parse the response.
        payload = r.json() or {}
        msgs = payload.get("messages", []) or []
        if not isinstance(msgs, list):
            return []
        return msgs  # each item is a dict

    def collect_raw(self, tickers: List[str], stem: str, run_id: str) -> Path | None:
        
        # dedupe, preserve order.
        tickers = list(dict.fromkeys(tickers))
        if not tickers:
            return None
        
        # create the raw stocktwits directory.
        raw_stocktwits_dir.mkdir(parents=True, exist_ok=True)

        # create the output path.
        out_path = raw_stocktwits_dir / f"stocktwits_raw_{stem}_{run_id}.jsonl"

        # write the messages to the output path.
        wrote = 0
        with out_path.open("w", encoding="utf-8") as f:
            for t in tickers:
                # fetch the messages for the ticker.
                try:
                    msgs = self.fetch_symbol_messages(t, stocktwits_limit_per_ticker)
                except Exception:
                    continue

                # write the messages to the output path.
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    # attach ticker so we don't rely on the message payload shape.
                    m = dict(m)
                    m["ticker"] = t
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
                    wrote += 1

        # print the result.
        print(f"{Fore.CYAN}stage 2.5 - saved stocktwits raw to {Style.RESET_ALL}{out_path.name} ({wrote} msgs)")
        return out_path


