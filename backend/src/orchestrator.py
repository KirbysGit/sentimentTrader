
import os
import sys
import logging
import pandas as pd
from typing import List
from pathlib import Path
from colorama import Fore, Style
from datetime import datetime, timezone


# ensure backend is importable.
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
sys.path.insert(0, str(backend_dir))

# pipeline stage imports.
from src.a_social.reddit_collector import RedditCollector
from src.b_analysis.reddit_processor import RedditProcessor
# from src.c_features.stocktwits_collector import StocktwitsCollector
from src.c_features.google_trends_collector import GoogleTrendsCollector
from src.d_stocks.stock_collector import StockCollector
from src.e_merge.feature_builder import FeatureBuilder
# from src.f_train.train_baseline import train_baseline

# util imports.
from src.utils.ticker_selection import grab_top_tickers
from src.utils.config import comment_weight

logger = logging.getLogger(__name__)


class PipelineOrchestrator:

    # --- self-initialize.

    def __init__(self):
        print(f"{Fore.CYAN}=== pipeline is ready! ===\n{Style.RESET_ALL}")

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
        self.stocktwits_messages_path = None
        self.stocktwits_daily_path = None
        self.trends_collector = GoogleTrendsCollector()
        self.trends_daily_path = None

        # -- 4. stock collector (single instance).
        self.stock_collector = StockCollector()
        self.stocks_by_ticker_dir = None
            
    # stage 1 - reddit collection.

    def collect_reddit_data(self):
        try:
            return self.reddit_collector.fetch_data()
        except Exception as e:
            print(f"{Fore.RED}stage 1 - uh oh : {e} {Style.RESET_ALL}")
            return None

    # stage 2 - process reddit data and set top tickers.
    def process_social_data(self, df: pd.DataFrame):
        try:
            # process reddit data.
            processed = self.reddit_processor.process(df=df, run_id=self.run_id)

            # check if processed data is valid.
            ok = processed is not None and not processed.empty

            if not ok:
                print(f"{Fore.RED}stage 2 - no data found after processing {Style.RESET_ALL}")
                return []

            # grab top tickers from daily metrics.
            self.post_processing_tickers = grab_top_tickers(processed=processed, run_id=self.run_id)

            print(f"\n{Fore.CYAN}--- after processing, we have these tickers ---{Style.RESET_ALL}")
            print(set(self.post_processing_tickers))

            return self.post_processing_tickers or []
        except Exception as e:
            print(f"{Fore.RED}stage 2 - uh oh : {e} {Style.RESET_ALL}")
            return []
            
    # stage 3 - grab data from stocktwits on relevant tickers.
    def source_features(self, tickers: List[str]):
        try:
            # -- 1. collect the google trends.
            res = self.trends_collector.collect(tickers=tickers, run_id=self.run_id)
            if res.ok and res.path and res.rows > 0:
                self.trends_daily_path = Path(res.path)
                print(
                    f"{Fore.CYAN}stage 3 - saved google trends to {Style.RESET_ALL}{Path(res.path).name} "
                    f"{Fore.CYAN}({res.rows} rows){Style.RESET_ALL}"
                )
            else:
                detail = res.error or "unknown"
                print(f"{Fore.YELLOW}stage 3 - google trends empty/skipped: {detail}{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}stage 3 - uh oh : {e} {Style.RESET_ALL}")
            return False

    # stage 4 - collecting relevant stock data.
    def collect_stock_data(self, tickers: List[str]):
        try:
            out_dir = self.stock_collector.collect_stock_data(tickers=tickers, run_id=self.run_id)
            self.stocks_by_ticker_dir = out_dir
            print(f"{Fore.CYAN}stage 4 - wrote per-ticker stock files to: {Style.RESET_ALL}{out_dir}")
            return True
        except Exception as e:
            print(f"{Fore.RED}stage 4 - uh oh : {e} {Style.RESET_ALL}")
            return False

    """
    # stage 5 - merge reddit daily metrics + stock OHLCV into one training dataset.
    def build_features(self):
        try:
            # check if stocks_by_ticker_dir is set.
            stocks_dir = self.stocks_by_ticker_dir
            if not stocks_dir:
                print(f"{Fore.RED}stage 5 - missing stocks_by_ticker_dir (run stage 4 first).{Style.RESET_ALL}")
                return False

            # build the features.
            out_path = self.feature_builder.build_dataset(
                raw_output_path=Path(self.raw_output_path),
                stocks_by_ticker_dir=Path(stocks_dir),
                tickers=self.post_processing_tickers,
            )

            # set the merged features path.
            self.merged_features_path = out_path

            # check if the merged features path exists.
            ok = out_path is not None and out_path.exists()

            # return the result.
            return bool(ok)
        except Exception as e:
            print(f"{Fore.RED}stage 4 - uh oh : {e} {Style.RESET_ALL}")
            return False

    # stage 6 - baseline training.
    def train_model(self):
        try:
            p = self.merged_features_path
            if not p:
                print(f"{Fore.RED}stage 5 - missing merged_features_path (run stage 4 first).{Style.RESET_ALL}")
                return False
            result = train_baseline(Path(p))
            return bool(result.ok and result.report_path.exists())
        except Exception as e:
            print(f"{Fore.RED}stage 5 - uh oh : {e} {Style.RESET_ALL}")
            return False
    """

# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def main():    
    orchestrator = PipelineOrchestrator()

    # phase 1 : collect reddit.
    df_new = orchestrator.collect_reddit_data()
    df_refresh = orchestrator.reddit_collector.refresh_recent_posts(days=7)

    # Combine "new" + "refreshed" and keep the snapshot with the highest engagement per post id.
    frames = [x for x in [df_new, df_refresh] if x is not None and not x.empty]
    df = pd.concat(frames, ignore_index=True) if frames else None
    if df is not None and not df.empty:
        try:
            df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0)
            df["num_comments"] = pd.to_numeric(df.get("num_comments", 0), errors="coerce").fillna(0)
            df["_engagement"] = df["score"] + (df["num_comments"] * float(comment_weight))
            # Keep highest engagement per post id
            df = (
                df.sort_values(["id", "_engagement"])
                .drop_duplicates(subset=["id"], keep="last")
                .drop(columns=["_engagement"], errors="ignore")
                .reset_index(drop=True)
            )
        except Exception:
            pass

        print(f"{Fore.CYAN}=== OK stage 1 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}=== no new reddit data collected. stopping pipeline (try again later). ===\n\n{Style.RESET_ALL}")
        print(f"{Fore.RED}=== FAIL stage 1 failed! ===\n{Style.RESET_ALL}")
        return

    # phase 2 : process reddit data.
    tickers = orchestrator.process_social_data(df=df)
    if tickers:
        print(f"{Fore.CYAN}=== OK stage 2 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== FAIL stage 2 failed! ===\n{Style.RESET_ALL}")
        return
    
    # phase 3 : source features.
    if orchestrator.source_features(tickers=tickers):
        print(f"{Fore.CYAN}=== OK stage 3 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== FAIL stage 3 failed! ===\n{Style.RESET_ALL}")
        return

    # phase 4 : collect stock data.
    if orchestrator.collect_stock_data(tickers=tickers):
        print(f"{Fore.CYAN}=== OK stage 4 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== FAIL stage 4 failed! ===\n{Style.RESET_ALL}")
        return

    return

    # for right now, we are in the data accumulation phase.
    # because we got kind of limited on our data sources from the past,
    # our best option is just to build and aggregate as time goes then
    # begin training once we have a good amount of data.

    """

    # phase 5 : build merged features.
    if orchestrator.build_features():
        print(f"{Fore.GREEN}=== OK stage 4 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== FAIL stage 4 failed! ===\n{Style.RESET_ALL}")
        return

    # phase 6 : baseline training.
    if orchestrator.train_model():
        print(f"{Fore.GREEN}=== OK stage 5 done! ===\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}=== FAIL stage 5 failed! ===\n{Style.RESET_ALL}")
        return

    """

if __name__ == "__main__":
    main() 
