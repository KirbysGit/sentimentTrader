"""
run stages after reddit (merge + optional train) using data already on disk.

from backend:
  python scripts/pipeline/run_pipeline_tail.py
  python scripts/pipeline/run_pipeline_tail.py --no-train
  python scripts/pipeline/run_pipeline_tail.py --refresh-stocks
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from colorama import Fore, Style

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from src.d_stocks.stock_collector import StockCollector  # noqa: E402
from src.e_merge.feature_builder import FeatureBuilder  # noqa: E402
from src.f_train.train_baseline import train_baseline  # noqa: E402
from src.utils.config import run_baseline_training_after_merge  # noqa: E402
from src.utils.path_config import raw_stocks_dir  # noqa: E402


def _tickers_from_existing_stock_files() -> list[str]:
    by_ticker = raw_stocks_dir / "by_ticker"
    out: list[str] = []
    if not by_ticker.is_dir():
        return out
    for p in sorted(by_ticker.glob("raw_*.csv")):
        stem = p.stem
        if stem.startswith("raw_"):
            out.append(stem[4:])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="merge sentiment + stocks without reddit processing.")
    parser.add_argument(
        "--refresh-stocks",
        action="store_true",
        help="yfinance refresh for every symbol that already has raw_<T>.csv",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="skip baseline training after merge",
    )
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    by_ticker_dir = raw_stocks_dir / "by_ticker"

    print(f"{Fore.CYAN}=== pipeline tail (no reddit) run_id={run_id} ==={Style.RESET_ALL}\n")

    if args.refresh_stocks:
        tickers = _tickers_from_existing_stock_files()
        if not tickers:
            print(f"{Fore.RED}no raw_*.csv under {by_ticker_dir}; nothing to refresh.{Style.RESET_ALL}")
            sys.exit(1)
        print(f"{Fore.CYAN}refreshing OHLCV for {len(tickers)} symbols...{Style.RESET_ALL}")
        StockCollector().collect_stock_data(tickers, run_id=run_id, apply_topn_limit=False)
        print(f"{Fore.GREEN}=== stock refresh done ==={Style.RESET_ALL}\n")

    if not by_ticker_dir.is_dir() or not any(by_ticker_dir.glob("raw_*.csv")):
        print(f"{Fore.RED}missing stock files under {by_ticker_dir}.{Style.RESET_ALL}")
        sys.exit(1)

    fb = FeatureBuilder()
    merged_path = fb.build_dataset(stocks_by_ticker_dir=by_ticker_dir, run_id=run_id, tickers=None)
    if not merged_path.exists():
        print(f"{Fore.RED}merge failed: {merged_path}{Style.RESET_ALL}")
        sys.exit(1)

    n = len(pd.read_csv(merged_path))
    print(f"{Fore.GREEN}=== merge done ({n:,} rows in {merged_path.name}) ==={Style.RESET_ALL}\n")

    do_train = run_baseline_training_after_merge and not args.no_train
    if not do_train:
        if args.no_train:
            print(f"{Fore.CYAN}skipped training (--no-train).{Style.RESET_ALL}")
        sys.exit(0)

    if n < 20:
        print(f"{Fore.YELLOW}skip training: need >= 20 merged rows, got {n}.{Style.RESET_ALL}")
        sys.exit(0)

    result = train_baseline(merged_path)
    if result.ok:
        print(f"{Fore.GREEN}=== training done ==={Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}training failed.{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
