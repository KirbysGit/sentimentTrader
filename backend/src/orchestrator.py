
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
from src.a_social.rd_collector import RedditCollector
from src.a_social.stocktwits_collector import StocktwitsCollector
from src.b_analysis.reddit_processor import RedditProcessor
from src.c_stocks.stock_collector import StockCollector
from src.d_merge.feature_builder import FeatureBuilder
from src.e_train.train_baseline import train_baseline

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

        # -- 2.75. optional stocktwits collector.
        self.stocktwits_collector = StocktwitsCollector()

        # -- 3. stage 3 collector (single instance).
        self.stock_collector = StockCollector()

        # -- 4. stage 3 output dir (set after stage 3 runs).
        self.stocks_by_ticker_dir = None

        # -- 5. stage 4 builder (single instance).
        self.feature_builder = FeatureBuilder()

        # -- 6. stage 4 output path (set after stage 4 runs).
        self.merged_features_path = None
            
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

    # stage 2.5 - optional stocktwits ingest (non-fatal)
    def collect_stocktwits(self):
        try:
            tickers = self.stage3_tickers or []
            if not tickers or not self.raw_output_path:
                return True
            stem = Path(self.raw_output_path).stem
            self.stocktwits_collector.collect(tickers=tickers, stem=stem, run_id=self.run_id)
            return True
        except Exception as e:
            print(f"{Fore.YELLOW}stage 2.5 - stocktwits skipped: {e}{Style.RESET_ALL}")
            return True
            
    # stage 3 - collecting relevant stock data.
    def collect_stock_data(self):
        try:
            tickers = self.stage3_tickers or []
            if not tickers:
                print(f"{Fore.RED}stage 3 - no tickers selected (watchlist empty).{Style.RESET_ALL}")
                return False

            out_dir = self.stock_collector.collect_stock_data(tickers=tickers)
            self.stocks_by_ticker_dir = out_dir
            print(f"{Fore.CYAN}stage 3 - wrote per-ticker stock files to: {Style.RESET_ALL}{out_dir}")
            return True
        except Exception as e:
            print(f"{Fore.RED}stage 3 - uh oh 🚨 : {e} {Style.RESET_ALL}")
            return False

    # stage 4 - merge reddit daily metrics + stock OHLCV into one training dataset.
    def build_features(self):
        try:
            if not self.raw_output_path:
                print(f"{Fore.RED}stage 4 - missing raw_output_path.{Style.RESET_ALL}")
                return False

            # check if stocks_by_ticker_dir is set.
            stocks_dir = self.stocks_by_ticker_dir
            if not stocks_dir:
                print(f"{Fore.RED}stage 4 - missing stocks_by_ticker_dir (run stage 3 first).{Style.RESET_ALL}")
                return False

            # build the features.
            out_path = self.feature_builder.build_dataset(
                raw_output_path=Path(self.raw_output_path),
                stocks_by_ticker_dir=Path(stocks_dir),
                tickers=self.stage3_tickers,
            )

            # set the merged features path.
            self.merged_features_path = out_path

            # check if the merged features path exists.
            ok = out_path is not None and out_path.exists()

            # return the result.
            return bool(ok)
        except Exception as e:
            print(f"{Fore.RED}stage 4 - uh oh 🚨 : {e} {Style.RESET_ALL}")
            return False

    # stage 5 - baseline training.
    def train_model(self):
        try:
            p = self.merged_features_path
            if not p:
                print(f"{Fore.RED}stage 5 - missing merged_features_path (run stage 4 first).{Style.RESET_ALL}")
                return False
            result = train_baseline(Path(p))
            return bool(result.ok and result.report_path.exists())
        except Exception as e:
            print(f"{Fore.RED}stage 5 - uh oh 🚨 : {e} {Style.RESET_ALL}")
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

    # phase 2.5 : optional stocktwits ingest (never blocks pipeline).
    orchestrator.collect_stocktwits()

    break

    # phase 3 : collect stock data.
    if orchestrator.collect_stock_data():
        print(f"{Fore.GREEN}=== ✓ stage 3 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== ✗ stage 3 failed! ===\n{Style.RESET_ALL}")
        return

    # phase 4 : build merged features.
    if orchestrator.build_features():
        print(f"{Fore.GREEN}=== ✓ stage 4 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== ✗ stage 4 failed! ===\n{Style.RESET_ALL}")
        return

    # phase 5 : baseline training.
    if orchestrator.train_model():
        print(f"{Fore.GREEN}=== ✓ stage 5 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== ✗ stage 5 failed! ===\n{Style.RESET_ALL}")
        return

if __name__ == "__main__":
    main() 
