
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
# Daily run: accumulation only (reddit → metrics → trends → OHLCV). Merge + train: scripts/pipeline or run manually.
# from src.e_merge.feature_builder import FeatureBuilder
# from src.f_train.train_baseline import train_baseline

# util imports.
from src.utils.ticker_selection import grab_top_tickers
from src.utils.config import comment_weight
# from src.utils.config import run_baseline_training_after_merge

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

        # -- 5. merge + train (disabled for daily cron; see scripts/pipeline/run_pipeline_tail.py + train_baseline).
        # self.feature_builder = FeatureBuilder()
        # self.merged_features_path = None

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

    # # stage 5 - merge reddit daily master + per-ticker OHLCV into training features.
    # def build_features(self, tickers: Optional[List[str]] = None):
    #     try:
    #         stocks_dir = self.stocks_by_ticker_dir
    #         if not stocks_dir:
    #             print(f"{Fore.RED}stage 5 - missing stocks_by_ticker_dir (run stage 4 first).{Style.RESET_ALL}")
    #             return False
    #
    #         # tickers=None builds rows for every symbol that appears in reddit_daily_all and has raw_{t}.csv.
    #         out_path = self.feature_builder.build_dataset(
    #             stocks_by_ticker_dir=Path(stocks_dir),
    #             run_id=self.run_id,
    #             tickers=tickers,
    #         )
    #         self.merged_features_path = out_path
    #         ok = out_path is not None and out_path.exists()
    #         return bool(ok)
    #     except Exception as e:
    #         print(f"{Fore.RED}stage 5 - uh oh : {e} {Style.RESET_ALL}")
    #         return False
    #
    # # stage 6 - baseline training (logistic regression on merged_features_all.csv).
    # def train_model(self):
    #     try:
    #         p = self.merged_features_path
    #         if not p or not Path(p).exists():
    #             print(f"{Fore.RED}stage 6 - missing merged_features_path (run stage 5 first).{Style.RESET_ALL}")
    #             return False
    #         check = pd.read_csv(p)
    #         if check.empty or len(check) < 20:
    #             print(
    #                 f"{Fore.YELLOW}stage 6 - skip training: merged file has {len(check)} rows (need >= 20).{Style.RESET_ALL}"
    #             )
    #             return False
    #         result = train_baseline(Path(p))
    #         return bool(result.ok and result.report_path.exists())
    #     except Exception as e:
    #         print(f"{Fore.RED}stage 6 - uh oh : {e} {Style.RESET_ALL}")
    #         return False

# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def main():    
    orchestrator = PipelineOrchestrator()

    # phase 1 : collect reddit.
    df_new = orchestrator.collect_reddit_data()
    df_refresh = orchestrator.reddit_collector.refresh_recent_posts(days=7)

    # combine new & refreshed reddit data.
    frames = [x for x in [df_new, df_refresh] if x is not None and not x.empty]
    
    df = pd.concat(frames, ignore_index=True) if frames else None
    if df is not None and not df.empty:
        try:
            # -- convert the score and num_comments to numeric.
            df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0)
            df["num_comments"] = pd.to_numeric(df.get("num_comments", 0), errors="coerce").fillna(0)
            df["_engagement"] = df["score"] + (df["num_comments"] * float(comment_weight))
            
            # -- keep highest engagement per post id.
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
    
    # -----
    
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

    # # phase 5–6 : merge + baseline train — run on demand: `python scripts/pipeline/run_pipeline_tail.py` (and train if desired).
    # if orchestrator.build_features(tickers=None):
    #     print(f"{Fore.GREEN}=== OK stage 5 done! ===\n{Style.RESET_ALL}")
    # else:
    #     print(f"{Fore.RED}=== FAIL stage 5 failed! ===\n{Style.RESET_ALL}")
    #     return
    #
    # if run_baseline_training_after_merge:
    #     if orchestrator.train_model():
    #         print(f"{Fore.GREEN}=== OK stage 6 done! ===\n{Style.RESET_ALL}")
    #     else:
    #         print(f"{Fore.YELLOW}=== stage 6 skipped or failed (see logs above) ===\n{Style.RESET_ALL}")

    print(f"{Fore.GREEN}=== daily accumulation complete (stages 1–4). ===\n{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 
