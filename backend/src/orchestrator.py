
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from colorama import Fore, Style


# ensure backend is importable.
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
sys.path.insert(0, str(backend_dir))

# pipeline stage imports.
from src.a_reddit.rd_collector import RedditCollector
from src.b_analysis.reddit_processor import RedditProcessor

logger = logging.getLogger(__name__)


class PipelineOrchestrator:

    # --- self-initialize.

    def __init__(self):
        print(f"{Fore.CYAN}=== pipeline 🤓 is ready!===\n{Style.RESET_ALL}")

        # -- 1. run start data.
        self.run_ts = datetime.now(timezone.utc)                            # run time.
        self.run_date = self.run_ts.date().isoformat()                      # run date.
        self.run_id = self.run_ts.strftime("%Y%m%d_%H%M%S")                 # run id.

        # -- 2. after phase 1 : our reddit dir
        self.raw_output_path = None                                             
            
    # stage 1 - reddit collection.

    def collect_reddit_data(self):
        try:
            collector = RedditCollector(run_date=self.run_date, run_id=self.run_id)
            
            output_path = collector.fetch_data()
            success = output_path is not None

            if success:
                self.raw_output_path = output_path
                return success
            else:
                return False
            
        except Exception as e:
            print(f"{Fore.RED}stage 1 - uh oh 🚨 : {e} {Style.RESET_ALL}")
            return False

    # stage 2 - process reddit data.

    def process_reddit_data(self):
        try:
            processor = RedditProcessor(input_file=self.raw_output_path)
            
            df = processor.process()

            # quick preview (first 3 rows) so we can sanity-check output shape.
            if df is not None and not df.empty:
                preview_cols = [
                    "created_at",
                    "subreddit",
                    "post_id",
                    "ticker",
                    "boost_score",
                    "sentiment_score",
                    "sentiment_category",
                    "score",
                    "num_comments",
                    "upvote_ratio",
                ]
                preview_cols = [c for c in preview_cols if c in df.columns]
                print(f"\n{Fore.CYAN}--- stage 2 preview (first 3 rows) ---{Style.RESET_ALL}")
                print(df[preview_cols].head(10).to_string(index=False))
            
            return df is not None and not df.empty
            
        except Exception as e:
            print(f"{Fore.RED}stage 1 - uh oh 🚨 : {e} {Style.RESET_ALL}")
            return False

    """
    # ------------------------------------------------------------
    # Stage 3 — Stock Data Collection
    # ------------------------------------------------------------
    def collect_stock_data(self, lookback_days=60):
        #Load processed Reddit output → determine which tickers → 
        #fetch historical stock data for those tickers.
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
        grouped = (
            df.groupby("ticker")
              .agg({"mean_confidence": "mean", "total_engagement": "sum"})
              .reset_index()
              .rename(columns={
                  "mean_confidence": "confidence",
                  "total_engagement": "engagement",
              })
        )

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
    """


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def main():    
    orchestrator = PipelineOrchestrator()

    # phase 1 : collect reddit.

    if orchestrator.collect_reddit_data():
        print(f"{Fore.CYAN}=== ✓ stage 1 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== ✗ stage 1 failed! ===\n{Style.RESET_ALL}")
        return

    # phase 2 : process reddit.
    if orchestrator.process_reddit_data():
        print(f"{Fore.CYAN}=== ✓ stage 2 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== ✗ stage 2 failed! ===\n{Style.RESET_ALL}")

    """
    # ---------------------------
    # Stage 3: Stock Data
    # ---------------------------
    if orchestrator.collect_stock_data(lookback_days=90):
        print(f"{Fore.GREEN}✓ Stage 3 completed successfully{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Stage 3 failed{Style.RESET_ALL}")
        return
    """

if __name__ == "__main__":
    main() 
