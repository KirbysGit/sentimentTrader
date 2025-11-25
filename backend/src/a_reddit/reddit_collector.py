# ===========================================================================
# reddit_collector.py — Stage 1 (Reddit Collection)
#
# PURPOSE:
#   Scrape high-signal Reddit subreddits, extract rich metadata optimized
#   for market prediction, and save a cleaned CSV for Stage 2.
#
# NEW ADDITIONS (PREDICTION-UPGRADED):
#   • Author metadata:
#       - author_name, author_karma, author_is_mod, author_created_utc
#
#   • Post metadata:
#       - domain (parsed from URL)
#       - post_hint (image, link, text, rich:video, etc.)
#       - is_self_post, is_image_post, is_link_post, is_video_post
#       - is_crosspost, crosspost_parent, crosspost_subreddit
#       - awards_count
#
#   • Comments metadata:
#       - top_comments: list of dicts
#           {
#             "body": "...",
#             "score": int,
#             "author": "...",
#             "author_flair": "..."
#           }
#
#   • Uses updated subreddit lists from config.py (must-have + optional).
#
# ORGANIZATION:
#   1. Imports + environment
#   2. RedditDataCollector class
#      - helpers for media detection, author info, domain parsing
#      - fetch_subreddit_posts
#      - fetch_all_subreddits
#
# ===========================================================================

import os
import logging
import pandas as pd
from praw import Reddit
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
from colorama import Fore, Style
from datetime import datetime, timedelta

from src.utils.path_config import RAW_DIR
from src.utils.config import (
    SUBREDDITS,
    SORT_METHODS,
    TEST_SUBREDDITS,
    TEST_SORT_METHODS,
)

# load environment vars
backend_dir = Path(__file__).parent.parent.parent
env_path = backend_dir / ".env"
load_dotenv(env_path if env_path.exists() else None)

logger = logging.getLogger(__name__)


# ===========================================================================
# RedditDataCollector
# ===========================================================================
class RedditDataCollector:

    def __init__(self, data_dir=None, max_days_lookback=30):
        self.data_dir = data_dir or (RAW_DIR / "reddit_data")
        self.max_days_lookback = max_days_lookback
        os.makedirs(self.data_dir, exist_ok=True)

        # initialize PRAW client
        self.reddit = Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="script:MarketSentiment:v1 (u/kiiiiiiiiirb)",
            refresh_token=os.getenv("REFRESH_TOKEN"),
        )

        # cutoff timestamp for filtering old posts
        self.cutoff_date = datetime.now() - timedelta(days=max_days_lookback)

    # =======================================================================
    # HELPER METHODS
    # =======================================================================

    def _select_time_filter(self):
        """Choose Reddit time_filter for 'top' sorting."""
        if self.max_days_lookback <= 7:
            return "week"
        if self.max_days_lookback <= 30:
            return "month"
        return "year"

    def _get_sort_stream(self, subreddit, sort, limit):
        """Return correct listing generator based on chosen sort method."""
        if sort == "hot":
            return subreddit.hot(limit=limit)
        if sort == "new":
            return subreddit.new(limit=limit)
        if sort == "top":
            return subreddit.top(time_filter=self._select_time_filter(), limit=limit)
        return []

    # -----------------------
    # Author metadata helper
    # -----------------------
    def _extract_author_info(self, post):
        """Extract author metadata safely (some posts have deleted authors)."""
        try:
            author = post.author
            if not author:
                return None, None, None, None

            return (
                author.name,
                getattr(author, "comment_karma", None) + getattr(author, "link_karma", None),
                author.is_mod,
                datetime.fromtimestamp(author.created_utc),
            )
        except:
            return None, None, None, None

    # -----------------------------
    # URL domain / media type helper
    # -----------------------------
    def _extract_post_media_info(self, post):
        """
        Classify post type & domain.
        Uses common Reddit post_hint attributes when available.
        """
        url = post.url or ""
        domain = urlparse(url).netloc or None

        post_hint = getattr(post, "post_hint", None)

        is_self = post.is_self
        is_image = post_hint in ["image"]
        is_link = (not is_self) and post_hint in ["link", "rich:link"]
        is_video = post_hint in ["hosted:video", "rich:video"]

        # crosspost data (optional)
        is_cross = hasattr(post, "crosspost_parent_list")
        cross_parent = None
        cross_subreddit = None

        if is_cross:
            try:
                cp = post.crosspost_parent_list[0]
                cross_parent = cp.get("id")
                cross_subreddit = cp.get("subreddit")
            except:
                pass

        awards = getattr(post, "total_awards_received", 0)

        return {
            "domain": domain,
            "post_hint": post_hint,
            "is_self_post": is_self,
            "is_image_post": is_image,
            "is_link_post": is_link,
            "is_video_post": is_video,
            "is_crosspost": is_cross,
            "crosspost_parent": cross_parent,
            "crosspost_subreddit": cross_subreddit,
            "awards_count": awards,
        }

    # -----------------------------
    # Enhanced comment extraction
    # -----------------------------
    def _get_top_comments(self, post, max_count=5):
        """
        Extract top comments with score & author info.
        Returns list of dicts.
        """
        try:
            post.comments.replace_more(limit=0)
        except:
            return []

        comments_out = []
        for c in post.comments[:max_count]:
            try:
                comments_out.append({
                    "body": c.body,
                    "score": getattr(c, "score", None),
                    "author": getattr(c.author, "name", None) if c.author else None,
                    "author_flair": getattr(c, "author_flair_text", None),
                })
            except:
                continue

        return comments_out

    # =======================================================================
    # FETCH POSTS FROM ONE SUBREDDIT
    # =======================================================================
    def fetch_subreddit_posts(self, subreddit_name, limit=100, sort="hot"):
        """
        Fetch posts from a single subreddit using a specific sort method.
        Returns a DataFrame or None.
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = self._get_sort_stream(subreddit, sort, limit)
            out_rows = []

            for post in posts:
                created = datetime.fromtimestamp(post.created_utc)
                if created < self.cutoff_date:
                    continue

                # author metadata
                author_name, author_karma, author_is_mod, author_created = \
                    self._extract_author_info(post)

                # post media metadata
                media_info = self._extract_post_media_info(post)

                # build structured row
                out_rows.append({
                    "id": post.id,
                    "title": post.title,
                    "text": post.selftext,
                    "created_utc": created,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "upvote_ratio": post.upvote_ratio,
                    "subreddit": subreddit_name,
                    "flair": post.link_flair_text,

                    # author metadata
                    "author_name": author_name,
                    "author_karma": author_karma,
                    "author_is_mod": author_is_mod,
                    "author_created_utc": author_created,

                    # media metadata
                    **media_info,

                    # enhanced comments
                    "top_comments": self._get_top_comments(post),
                })

            return pd.DataFrame(out_rows) if out_rows else None

        except Exception as e:
            logger.error(f"error fetching r/{subreddit_name}: {e}")
            return None

    # =======================================================================
    # FETCH ALL SUBREDDITS + SAVE CSV
    # =======================================================================
    def fetch_all_subreddits(self, limit=50, test_mode=False):
        print(f"{Fore.CYAN}===== Stage 1: Reddit Data Collection ====={Style.RESET_ALL}")
        print(f"lookback={self.max_days_lookback} days | limit={limit}\n")

        subreddits = TEST_SUBREDDITS if test_mode else SUBREDDITS
        sorts = TEST_SORT_METHODS if test_mode else SORT_METHODS

        all_dfs = []
        total = len(subreddits) * len(sorts)
        step = 0

        for name in subreddits:
            print(f"{Fore.YELLOW}r/{name}:{Style.RESET_ALL}")

            for sort in sorts:
                step += 1
                print(f"  [{step}/{total}] {sort}...", end="", flush=True)

                df = self.fetch_subreddit_posts(name, limit=limit, sort=sort)

                if df is not None and not df.empty:
                    all_dfs.append(df)
                    print(f"{Fore.GREEN} ✓ {len(df)} posts{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED} ✗{Style.RESET_ALL}")

            print()

        if not all_dfs:
            print(f"{Fore.RED}✗ no posts collected{Style.RESET_ALL}")
            return None

        final = (
            pd.concat(all_dfs, ignore_index=True)
            .drop_duplicates(subset=["id"])
            .sort_values("created_utc", ascending=False)
        )

        filename = f"reddit_posts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        final.to_csv(self.data_dir / filename, index=False)

        print(f"{Fore.GREEN}✓ collected {len(final)} total posts{Style.RESET_ALL}")
        print(f"  saved to: {filename}\n")

        return final
