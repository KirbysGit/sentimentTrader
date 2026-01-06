
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
from src.c_stocks.stock_collector import StockCollector

# util imports.
from src.utils.ticker_selection import grab_top_tickers

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

        # -- 2.5. after phase 2 : our stage 3 tickers passed on.                                             
        self.stage3_tickers = []

        # -- 3. stage 3 collector (single instance).
        self.stock_collector = StockCollector()
            
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

            # -- 1. grab top tickers from daily metrics.
            if df is not None and not df.empty:
                tickers = grab_top_tickers(self.raw_output_path)
                self.stage3_tickers = tickers
                print(f"\n{Fore.CYAN}--- stage 3 prep (selected tickers) ---{Style.RESET_ALL}")
                print(tickers)

            # -- 2. return success.
            return df is not None and not df.empty
            
        except Exception as e:
            print(f"{Fore.RED}stage 2 - uh oh 🚨 : {e} {Style.RESET_ALL}")
            return False

    # stage 3 - collecting relevant stock data.
    def collect_stock_data(self):
        try:
            tickers = self.stage3_tickers or []
            if not tickers:
                print(f"{Fore.RED}stage 3 - no tickers selected (watchlist empty).{Style.RESET_ALL}")
                return False

            tickers_path = self.stock_collector.collect_stock_data(tickers=tickers)
            print(f"{Fore.CYAN}stage 3 - wrote tickers list: {Style.RESET_ALL}{tickers_path.name}")
            return True
        except Exception as e:
            print(f"{Fore.RED}stage 3 - uh oh 🚨 : {e} {Style.RESET_ALL}")
            return False


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


    # phase 3 : collect stock data.
    if orchestrator.collect_stock_data():
        print(f"{Fore.GREEN}=== ✓ stage 3 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== ✗ stage 3 failed! ===\n{Style.RESET_ALL}")
        return

if __name__ == "__main__":
    main() 
