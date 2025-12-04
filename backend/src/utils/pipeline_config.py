# pipeline execution configuration
# adjust these values to control how the pipeline runs

# ============================================================================
# mode selection
# ============================================================================
# set to True for quick testing, False for full production collection
TEST_MODE = False

# ============================================================================
# test mode settings (fast, minimal data for development)
# ============================================================================
TEST_DAYS_LOOKBACK = 7  # look back 7 days
TEST_POSTS_PER_SUBREDDIT = 10  # 10 posts per subreddit/sort (triggers test_mode in collector)

# ============================================================================
# production mode settings (full data collection for sentiment analysis)
# ============================================================================
PROD_DAYS_LOOKBACK = 30  # look back 30 days for more historical data
PROD_POSTS_PER_SUBREDDIT = 100  # 100 posts per subreddit/sort method

# ============================================================================
# active settings (automatically selected based on TEST_MODE)
# ============================================================================
REDDIT_DAYS_LOOKBACK = TEST_DAYS_LOOKBACK if TEST_MODE else PROD_DAYS_LOOKBACK
REDDIT_POSTS_PER_SUBREDDIT = TEST_POSTS_PER_SUBREDDIT if TEST_MODE else PROD_POSTS_PER_SUBREDDIT
