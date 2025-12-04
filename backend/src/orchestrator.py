"""
Simple Pipeline Orchestrator
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from colorama import Fore, Style

# Suppress TF warnings (if TF ever gets used)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure backend is importable
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
sys.path.insert(0, str(backend_dir))

# Pipeline stage imports
from src.a_reddit.reddit_collector import RedditDataCollector
from src.b_analysis.reddit_data_processor import RedditDataProcessor

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self):
        print(f"{Fore.CYAN}=== Pipeline Orchestrator ==={Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Ready\n{Style.RESET_ALL}")
        self.run_ts = datetime.now(timezone.utc)
        self.run_date = self.run_ts.date().isoformat()
        self.run_id = self.run_ts.strftime("%Y%m%d_%H%M%S")
        self.raw_output_paths = []

    # ------------------------------------------------------------
    # Stage 1 — Reddit Collection
    # ------------------------------------------------------------
    def collect_reddit_data(self, days=30, limit=100):
        try:
            collector = RedditDataCollector(
                max_days_lookback=days,
                run_date=self.run_date,
                run_id=self.run_id,
            )
            test_mode = limit <= 20

            output_path = collector.fetch_all_subreddits(
                limit=limit,
                test_mode=test_mode,
            )

            success = output_path is not None
            if success:
                self.raw_output_paths = [output_path]
            return success
            
        except Exception as e:
            logger.error(f"collection error: {e}")
            print(f"{Fore.RED}✗ collection error: {e}{Style.RESET_ALL}")
            return False

    # ------------------------------------------------------------
    # Stage 2 — Clean + Process Reddit Data
    # ------------------------------------------------------------
    def process_reddit_data(self):
        try:
            processor = RedditDataProcessor(
                input_files=self.raw_output_paths,
                run_date=self.run_date,
                run_id=self.run_id,
            )
            df = processor.process()                                # <— only method now

            return df is not None and not df.empty
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"processing error: {e}\n{tb}")
            print(f"{Fore.RED}✗ processing error: {e}{Style.RESET_ALL}")
            # show file and line where error occurred
            for line in tb.split('\n'):
                if 'File "' in line and 'reddit_data_processor' in line:
                    print(f"{Fore.YELLOW}  {line.strip()}{Style.RESET_ALL}")
            return False

    # ------------------------------------------------------------
    # Stage 3 — Stock Data Collection
    # ------------------------------------------------------------
    def collect_stock_data(self, lookback_days=60):
        """
        Load processed Reddit output → determine which tickers → 
        fetch historical stock data for those tickers.
        """
        from datetime import datetime, timedelta
        from src.c_stocks.stock_data_collector import StockDataCollector
        from src.utils.path_config import PROCESSED_METRICS_DIR

        processed_file = PROCESSED_METRICS_DIR / "ticker_daily_metrics.csv"
        if not processed_file.exists():
            print(Fore.RED + "✗ No ticker_daily_metrics.csv found. Cannot collect stock data." + Style.RESET_ALL)
            return False
            
        import pandas as pd
        df = pd.read_csv(processed_file)
        if df.empty:
            print(Fore.RED + "✗ ticker_daily_metrics.csv is empty." + Style.RESET_ALL)
            return False

        # --- Determine which tickers to fetch ---
        grouped = df.groupby("ticker").agg({
            "confidence": "mean",
            "engagement": "sum",
        }).reset_index()

        # Filter rules
        filtered = grouped[grouped["confidence"] >= 0.5]
        tickers = filtered["ticker"].tolist()

        if not tickers:
            print(Fore.YELLOW + "⚠ No tickers passed the confidence filter. Skipping stock collection." + Style.RESET_ALL)
            return False

        # --- Configure date range ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # --- Initialize collector ---
        collector = StockDataCollector()
        collector.configure(
            symbols=tickers,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        # --- Collect historical stock data ---
        stock_data = collector.collect_data()
        if not stock_data:
            print(Fore.RED + "✗ No stock data collected." + Style.RESET_ALL)
            return False

        print(Fore.GREEN + f"✓ stock data collection: {len(stock_data)} tickers" + Style.RESET_ALL)
        return True



# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def main():
    from src.utils.pipeline_config import (
        REDDIT_DAYS_LOOKBACK,
        REDDIT_POSTS_PER_SUBREDDIT,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    orchestrator = PipelineOrchestrator()

    # ---------------------------
    # Stage 1: Collect Reddit
    # ---------------------------
    if orchestrator.collect_reddit_data(
        days=REDDIT_DAYS_LOOKBACK,
        limit=REDDIT_POSTS_PER_SUBREDDIT,
    ):
        print(f"{Fore.GREEN}✓ Stage 1 completed successfully{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Stage 1 failed{Style.RESET_ALL}")
        return

    # ---------------------------
    # Stage 2: Process Reddit
    # ---------------------------
    if orchestrator.process_reddit_data():
        print(f"{Fore.GREEN}✓ Stage 2 completed successfully{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Stage 2 failed{Style.RESET_ALL}")

    # ---------------------------
    # Stage 3: Stock Data
    # ---------------------------
    if orchestrator.collect_stock_data(lookback_days=90):
        print(f"{Fore.GREEN}✓ Stage 3 completed successfully{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Stage 3 failed{Style.RESET_ALL}")
        return


if __name__ == "__main__":
    main() 
