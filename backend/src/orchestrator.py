"""
Simple Pipeline Orchestrator
"""

# imports.
import os
import sys
import logging
from pathlib import Path
from colorama import Fore, Style

# suppress tensorflow warnings (if TF is even used)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ensure /backend is importable
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
sys.path.insert(0, str(backend_dir))

# clean, static imports
from src.a_reddit.reddit_collector import RedditDataCollector
from src.b_analysis.reddit_data_processor import RedditDataProcessor

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self):
        print(f"{Fore.CYAN}=== Pipeline Orchestrator ==={Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Ready\n{Style.RESET_ALL}")

    # ------------------------
    # stage 1
    # ------------------------
    def collect_reddit_data(self, days=30, limit=100):
        try:
            collector = RedditDataCollector(max_days_lookback=days)
            test_mode = limit <= 20

            df = collector.fetch_all_subreddits(
                limit=limit,
                test_mode=test_mode,
            )

            return df is not None and not df.empty

        except Exception as e:
            logger.error(f"collection error: {e}")
            print(f"{Fore.RED}✗ collection error: {e}{Style.RESET_ALL}")
            return False

    # ------------------------
    # stage 2
    # ------------------------
    def process_reddit_data(self):
        try:
            processor = RedditDataProcessor()
            df = processor.load_all_reddit_data()

            if df is None or df.empty:
                return False

            processed, daily = processor.process_reddit_data(df)
            return processed is not None and daily is not None

        except Exception as e:
            logger.error(f"processing error: {e}")
            print(f"{Fore.RED}✗ processing error: {e}{Style.RESET_ALL}")
            return False


# ------------------------
# CLI entrypoint
# ------------------------
def main():
    # import pipeline config.
    from src.utils.pipeline_config import REDDIT_DAYS_LOOKBACK, REDDIT_POSTS_PER_SUBREDDIT

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    orchestrator = PipelineOrchestrator()

    # stage 1: collect reddit data
    if orchestrator.collect_reddit_data(days=REDDIT_DAYS_LOOKBACK, limit=REDDIT_POSTS_PER_SUBREDDIT):
        print(f"{Fore.GREEN}✓ Stage 1 completed successfully{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Stage 1 failed{Style.RESET_ALL}")

    # stage 2: processing (disabled for now)
    # if orchestrator.process_reddit_data():
    #     print(f"{Fore.GREEN}✓ Pipeline completed successfully{Style.RESET_ALL}")
    # else:
    #     print(f"{Fore.RED}✗ Stage 2 failed{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
