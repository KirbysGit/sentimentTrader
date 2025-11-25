"""
Simple Pipeline Orchestrator
"""

import os
import sys
import logging
from pathlib import Path
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

    # ------------------------------------------------------------
    # Stage 1 — Reddit Collection
    # ------------------------------------------------------------
    def collect_reddit_data(self, days=30, limit=100):
        try:
            collector = RedditDataCollector(max_days_lookback=days)
            test_mode = limit <= 20

            df = collector.fetch_all_subreddits(
                limit=limit,
                test_mode=test_mode,
            )

            success = df is not None and not df.empty
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
            processor = RedditDataProcessor()
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
# CLI entrypoint
# ------------------------------------------------------------
def main():
        from src.utils.pipeline_config import (
            REDDIT_DAYS_LOOKBACK, 
            REDDIT_POSTS_PER_SUBREDDIT
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        orchestrator = PipelineOrchestrator()

        # ---------------------------
        # Stage 1: Collect Reddit
        # ---------------------------
        if orchestrator.collect_reddit_data(
            days=REDDIT_DAYS_LOOKBACK,
            limit=REDDIT_POSTS_PER_SUBREDDIT
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


if __name__ == "__main__":
    main()
