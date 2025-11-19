"""
Simple Pipeline Orchestrator - Reddit Data Collection Test
"""

import logging
import importlib
from colorama import Fore, Style

# Import Reddit collector
reddit_collector_module = importlib.import_module('src.1reddit.reddit_collector')
RedditDataCollector = reddit_collector_module.RedditDataCollector

# Import path config
from src.utils.path_config import REDDIT_DATA_DIR

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(self):
        """Initialize the pipeline orchestrator."""
        print(f"\n{Fore.CYAN}=== Reddit Data Collection Test ==={Style.RESET_ALL}")
        self.reddit_collector = RedditDataCollector()
        print(f"{Fore.GREEN}✓ Reddit collector initialized{Style.RESET_ALL}")
    
    def collect_reddit_data(self, 
                           max_days_lookback=30,
                           posts_per_subreddit=100):
        """Collect Reddit data.
        
        Args:
            max_days_lookback: Number of days to look back for Reddit posts
            posts_per_subreddit: Number of posts to collect per subreddit
        """
        try:
            print(f"\n{Fore.CYAN}Collecting Reddit Data...{Style.RESET_ALL}")
            print(f"• Lookback: {max_days_lookback} days")
            print(f"• Posts per subreddit: {posts_per_subreddit}")
            
            # Initialize collector with lookback period
            self.reddit_collector = RedditDataCollector(max_days_lookback=max_days_lookback)
            
            # Collect Reddit data
            reddit_data = self.reddit_collector.fetch_all_subreddits(limit=posts_per_subreddit)
            
            if reddit_data is None or reddit_data.empty:
                print(f"{Fore.RED}✗ Failed to collect Reddit data{Style.RESET_ALL}")
                return None
            
            print(f"{Fore.GREEN}✓ Collected {len(reddit_data)} posts from Reddit{Style.RESET_ALL}")
            return reddit_data
            
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {str(e)}")
            print(f"{Fore.RED}✗ Error: {str(e)}{Style.RESET_ALL}")
            return None

def main():
    """Main function to test Reddit data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Reddit Data Collection')
    parser.add_argument('--days', type=int, default=30, help='Days to look back for Reddit posts')
    parser.add_argument('--limit', type=int, default=100, help='Posts per subreddit')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and collect
    orchestrator = PipelineOrchestrator()
    reddit_data = orchestrator.collect_reddit_data(
        max_days_lookback=args.days,
        posts_per_subreddit=args.limit
    )
    
    if reddit_data is not None:
        print(f"\n{Fore.GREEN}✓ Reddit data collection test completed successfully{Style.RESET_ALL}")
        print(f"Data saved to: {REDDIT_DATA_DIR}")
    else:
        print(f"\n{Fore.RED}✗ Reddit data collection test failed{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
