
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
from src.a_social.reddit_collector import RedditCollector
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

        # -- pre-run start data.
        self.run_ts = datetime.now(timezone.utc)
        self.run_date = self.run_ts.date().isoformat()
        self.run_id = self.run_ts.strftime("%Y%m%d_%H%M%S")

        # -- 1. reddit collector.
        self.reddit_collector = RedditCollector(run_date=self.run_date, run_id=self.run_id)                                    

        # -- 2. reddit processor.
        self.reddit_processor = RedditProcessor()
        self.post_processing_tickers = []

        # -- 3. stocktwits collector. (feature sourcer)
        self.stocktwits_collector = StocktwitsCollector()
        self.stocktwits_messages_path = None
        self.stocktwits_daily_path = None

        # -- 4. stock collector (single instance).
        self.stock_collector = StockCollector()
        self.stocks_by_ticker_dir = None

        # -- 5. feature builder (single instance).
        self.feature_builder = FeatureBuilder()
        self.merged_features_path = None
            
    # stage 1 - reddit collection.

    def collect_reddit_data(self):
        try:
            return self.reddit_collector.fetch_data()
        except Exception as e:
            print(f"{Fore.RED}stage 1 - uh oh 🚨 : {e} {Style.RESET_ALL}")
            return None

    # stage 2 - process reddit data and set top tickers.
    def process_social_data(self, df: pd.DataFrame):
        try:
            processed = self.reddit_processor.process(df=df)

            ok = processed is not None and not processed.empty

            if not ok:
                print(f"{Fore.RED}stage 2 - no data found after processing 😡{Style.RESET_ALL}")
                return None

            # -- 2. grab top tickers from daily metrics.
            self.post_processing_tickers = grab_top_tickers(processed=processed)

            print(f"\n{Fore.CYAN}--- after processing, we have these tickers ---{Style.RESET_ALL}")
            print(set(self.post_processing_tickers))

            return True
            
    # stage 3 - grab data from stocktwits on relevant tickers.
    def source_features(self):
        try:
            tickers = self.post_processing_tickers or []
            if tickers and self.raw_output_path:
                stem = Path(self.raw_output_path).stem
                msgs_path, daily_path = self.stocktwits_collector.collect_and_process(
                    tickers=tickers,
                    stem=stem,
                    run_id=self.run_id,
                )
                self.stocktwits_messages_path = msgs_path
                self.stocktwits_daily_path = daily_path
                return True
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
        df = orchestrator.reddit_collector.fetch_data()
        print(f"{Fore.CYAN}=== ✓ stage 1 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}=== no new reddit data collected. stopping pipeline (try again later). ===\n\n{Style.RESET_ALL}")
        print(f"{Fore.RED}=== ✗ stage 1 failed! ===\n{Style.RESET_ALL}")
        return

    # phase 2 : process reddit data.
    if orchestrator.process_social_data(df=df):
        print(f"{Fore.CYAN}=== ✓ stage 2 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== ✗ stage 2 failed! ===\n{Style.RESET_ALL}")
        return
    
    # phase 3 : source features from stocktwits.
    if orchestrator.source_features():
        print(f"{Fore.CYAN}=== ✓ stage 3 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== ✗ stage 3 failed! ===\n{Style.RESET_ALL}")
        return

    # phase 4 : collect stock data.
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
