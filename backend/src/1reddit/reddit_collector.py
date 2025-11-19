# src / data_collection / reddit_collector.py

# Description : This file contains the RedditDataCollector class, which is used to collect data from Reddit.

# Imports.
import os
import logging
import pandas as pd
from praw import Reddit
from datetime import datetime, timedelta
from src.utils.path_config import RAW_DIR
from colorama import init, Fore, Style

# Setup Logging.
logger = logging.getLogger(__name__)

# Constants
TICKERS = ['NVDA', 'NVIDIA', 'AMD', 'INTC', 'TSMC']
SUBREDDITS = {
    'wallstreetbets': ['DD', 'Discussion', 'News'],
    'stocks': ['DD', 'Discussion', 'News'],
    'investing': ['Discussion', 'News'],
    'nvidia': ['Discussion', 'News', 'Rumor'],
    'AMD_Stock': ['Discussion', 'News'],
    'StockMarket': ['Discussion', 'News']
}

# Reddit Data Collector Class.
class RedditDataCollector:

    # -----------------------------------------------------------------------------------------------

    # Initialize Reddit Data Collector.
    def __init__(self, data_dir=None, max_days_lookback=30):
        """Initialize the Reddit Data Collector.
        
        Args:
            data_dir: Directory to store Reddit data
            max_days_lookback: Maximum number of days to look back for posts
        """
        self.data_dir = data_dir or (RAW_DIR / "reddit_data")
        self.max_days_lookback = max_days_lookback
        os.makedirs(self.data_dir, exist_ok=True)
        self.reddit = self._init_reddit()
        
    # -----------------------------------------------------------------------------------------------

    # Initialize Reddit Client.
    def _init_reddit(self):
        """Initialize Reddit Client with Refresh Token."""
        
        return Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent="script:Market Sentiment Analysis:v1.0 (by /u/kiiiiiiiiirb)",
            refresh_token=os.getenv('REFRESH_TOKEN')
        )
    
    # -----------------------------------------------------------------------------------------------
    
    def fetch_subreddit_posts(self, subreddit_name, limit=100, sort='hot', time_filter='day'):
        """
        Fetch Posts from a Subreddit with enhanced filtering.
        
        Args:
            subreddit_name (str): Name of Subreddit
            limit (int): Number of Posts to Fetch
            sort (str): Sort Method ('hot', 'new', 'top', 'relevance')
            time_filter (str): Time period to search ('day', 'week', 'month', 'year', 'all')
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_data = []
            cutoff_date = datetime.now() - timedelta(days=self.max_days_lookback)
            
            # Adjust time_filter based on lookback period
            if self.max_days_lookback <= 7:
                time_filter = 'week'
            elif self.max_days_lookback <= 30:
                time_filter = 'month'
            else:
                time_filter = 'year'
            
            # Get posts based on sort method
            if sort == 'hot':
                posts = subreddit.hot(limit=limit)
            elif sort == 'new':
                posts = subreddit.new(limit=limit)
            elif sort == 'top':
                posts = subreddit.top(time_filter=time_filter, limit=limit)
            
            # Process each post
            skipped_old = 0
            for post in posts:
                post_date = datetime.fromtimestamp(post.created_utc)
                
                # Skip if post is too old
                if post_date < cutoff_date:
                    skipped_old += 1
                    continue
                
                # Replace More Comments
                post.comments.replace_more(limit=0)
                
                # Get top comments
                top_comments = list(post.comments)[:5]
                
                # Create post data dictionary
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post_date,
                    'url': post.url,
                    'upvote_ratio': post.upvote_ratio,
                    'top_comments': [c.body for c in top_comments],
                    'subreddit': subreddit_name,
                    'flair': post.link_flair_text
                }
                
                posts_data.append(post_data)
            
            # Log collection statistics
            logger.info(f"r/{subreddit_name} ({sort}): Collected {len(posts_data)} posts, skipped {skipped_old} old posts")
            
            # Save to CSV
            if posts_data:
                df = pd.DataFrame(posts_data)
                filename = f"{subreddit_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                filepath = os.path.join(self.data_dir, filename)
                df.to_csv(filepath, index=False)
                logger.info(f"Saved to {filename} (date range: {df['created_utc'].min()} to {df['created_utc'].max()})")
            
            return pd.DataFrame(posts_data) if posts_data else None
            
        except Exception as e:
            logger.error(f"Error fetching data from r/{subreddit_name}: {str(e)}")
            return None
        
    # -----------------------------------------------------------------------------------------------

    # Fetch Post Comments.
    def fetch_post_comments(self, post_id, limit=None):
        """Fetch Comments for a Specific Post."""
        try:

            # Initialize Submission.
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=None)
            
            # Initialize Comments Data List.
            comments_data = []

            # Get Comments for Each Post.
            for comment in submission.comments.list():
                comment_data = {
                    'id': comment.id,
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc)
                }
                comments_data.append(comment_data)
                
                if limit and len(comments_data) >= limit:
                    break
            
            # Return Comments Data as DataFrame.
            return pd.DataFrame(comments_data)
            
        except Exception as e:
            logger.error(f"Error Fetching Comments for Post {post_id}: {str(e)}")
            return None

    # -----------------------------------------------------------------------------------------------

    def fetch_all_subreddits(self, limit=50):
        """Fetch recent posts from all monitored subreddits."""
        all_posts = []
        subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
        sort_methods = ['new', 'top', 'hot']
        
        total_steps = len(subreddits) * len(sort_methods)
        current_step = 0
        
        print(f"\n{Fore.CYAN}Reddit Data Collection Progress (lookback: {self.max_days_lookback} days):{Style.RESET_ALL}")
        print(f"Target: {limit} posts per subreddit/sort method\n")
        
        # Fetch from each subreddit
        for subreddit_name in subreddits:
            subreddit_total = 0
            print(f"{Fore.YELLOW}Processing r/{subreddit_name}:{Style.RESET_ALL}")
            
            # Try different sort methods to maximize coverage
            for sort in sort_methods:
                current_step += 1
                print(f"  [{current_step}/{total_steps}] Fetching {sort} posts...", end='', flush=True)
                
                posts = self.fetch_subreddit_posts(
                    subreddit_name,
                    limit=limit,
                    sort=sort,
                    time_filter='month' if self.max_days_lookback > 7 else 'week'
                )
                
                if posts is not None and not posts.empty:
                    all_posts.append(posts)
                    subreddit_total += len(posts)
                    print(f"{Fore.GREEN} ✓ {len(posts)} posts collected{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED} ✗ failed{Style.RESET_ALL}")
            
            print(f"  Total for r/{subreddit_name}: {Fore.CYAN}{subreddit_total} posts{Style.RESET_ALL}\n")
        
        # Combine all posts
        if all_posts:
            combined_df = pd.concat(all_posts, ignore_index=True)
            original_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['id'])
            duplicate_count = original_count - len(combined_df)
            
            # Sort by date and print summary
            combined_df = combined_df.sort_values('created_utc', ascending=False)
            
            print(f"{Fore.GREEN}Collection Summary:{Style.RESET_ALL}")
            print(f"• Total Posts: {len(combined_df):,}")
            print(f"• Duplicates Removed: {duplicate_count:,}")
            print(f"• Date Range: {combined_df['created_utc'].min()} to {combined_df['created_utc'].max()}")
            print(f"• Posts per Subreddit:")
            for subreddit in combined_df['subreddit'].unique():
                count = len(combined_df[combined_df['subreddit'] == subreddit])
                print(f"  - r/{subreddit}: {count:,}")
            
            return combined_df
        else:
            print(f"\n{Fore.RED}No posts were collected.{Style.RESET_ALL}")
            return None

def main():
    """Main function to test the RedditDataCollector."""
    import argparse
    init()
    
    parser = argparse.ArgumentParser(description='Collect Reddit data for stock analysis')
    parser.add_argument('--days', type=int, default=30, help='Number of days to look back')
    parser.add_argument('--limit', type=int, default=50, help='Posts per subreddit/sort method')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Initialize collector with lookback period
    collector = RedditDataCollector(max_days_lookback=args.days)

    # Collect posts
    df = collector.fetch_all_subreddits(limit=args.limit)

    if df is not None:
        print(f"\n{Fore.GREEN}Collection Complete:{Style.RESET_ALL}")
        print(f"• Total Posts: {len(df)}")
        print(f"• Date Range: {df['created_utc'].min()} to {df['created_utc'].max()}")
        print(f"• Posts per Subreddit:")
        for subreddit in df['subreddit'].unique():
            count = len(df[df['subreddit'] == subreddit])
            print(f"  - r/{subreddit}: {count}")
    else:
        print(f"{Fore.RED}No posts were collected.{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 