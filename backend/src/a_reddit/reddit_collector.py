# ===========================================================================
# reddit_collector.py — stage 1 (reddit data collection)
# ===========================================================================

"""
purpose:
  scrape high-signal Reddit subreddits, extract rich metadata optimized
  for market prediction, and save a cleaned CSV to pass onto stage 2.

organization:
  - imports
  - reddit data collector class
    - helper methods:
      - _select_time_filter
      - _get_sort_stream
      - _extract_author_info
      - _get_top_comments

    - fetch_subreddit_posts
    - fetch_all_subreddits

what we get:
  - author metadata: 
    - name
    - karma
    - is_mod
    - created_utc
  - post metadata: 
    - domain
    - post_hint
    - is_self_post
    - is_image_post
    - is_link_post
    - is_video_post
    - is_crosspost
    - crosspost_parent
    - crosspost_subreddit
    - awards_count
  - comments metadata: 
    - body
    - score
    - author
    - flair
"""
# imports.
import os
import json
import logging
import pandas as pd
from praw import Reddit
from pathlib import Path
from dotenv import load_dotenv
from colorama import Fore, Style
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone


# local imports.
from src.utils.path_config import RAW_REDDIT_DIR, PROCESSED_REDDIT_BY_DAY_DIR
from src.utils.config import (
    SUBREDDITS,
    SORT_METHODS,
    TEST_SUBREDDITS,
    TEST_SORT_METHODS,
)

from src.utils.pipeline_config import (
    REDDIT_DAYS_LOOKBACK,
    REDDIT_POSTS_PER_SUBREDDIT,
)

# load environment vars.
backend_dir = Path(__file__).parent.parent.parent
env_path = backend_dir / ".env"
load_dotenv(env_path if env_path.exists() else None)

# setup logging.
logger = logging.getLogger(__name__)


# ===========================================================================
# reddit data collector class
# ===========================================================================
class RedditDataCollector:

    def __init__(self, run_date: str | None = None, run_id: str | None = None):
        self._run_ts = datetime.now(timezone.utc)                                                       # run timestamp.
        self.run_date = run_date or self._run_ts.date().isoformat()                                     # run date.
        self.run_id = run_id or self._run_ts.strftime("%Y%m%d_%H%M%S")                                  # run id.
        self.data_dir = self._build_day_dir(RAW_REDDIT_DIR, self.run_date)                              # data directory for raw data output.
        self.max_days_lookback = REDDIT_DAYS_LOOKBACK                                                   # max # of days to look back for data.
        
        os.makedirs(self.data_dir, exist_ok=True)                                                       # create data directory if it doesn't exist.

        self.seen_registry_path = PROCESSED_REDDIT_BY_DAY_DIR.parent / "seen_post_ids.json"             # path to seen post ids registry.
        self._seen_ids = self._load_seen_ids()                                                          # load seen ids.

        # initialize PRAW client.
        self.reddit = Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="script:MarketSentiment:v1 (u/kiiiiiiiiirb)",
            refresh_token=os.getenv("REFRESH_TOKEN"),
        )

        # cutoff timestamp for filtering old posts.
        self.cutoff_date = datetime.now() - timedelta(days=REDDIT_DAYS_LOOKBACK)
        self.collected_data_path = None

    @staticmethod
    def _build_day_dir(root: Path, run_date: str) -> Path:
        """return yyyy/mm/dd folder under the given root."""
        try:
            year, month, day = run_date.split("-")
        except ValueError:
            raise ValueError(f"run_date must be yyyy-mm-dd, got {run_date}")
        return root / year / month / day

    # =======================================================================
    # helper methods
    # =======================================================================

    def _select_time_filter(self):
        """choose reddit time_filter for 'top' sort."""
        if self.max_days_lookback <= 7:
            return "week"
        if self.max_days_lookback <= 30:
            return "month"
        return "year"

    def _get_sort_stream(self, subreddit, sort, limit):
        """return correct listing generator based on chosen sort method."""
        if sort == "hot":
            return subreddit.hot(limit=limit)
        if sort == "new":
            return subreddit.new(limit=limit)
        if sort == "top":
            return subreddit.top(time_filter=self._select_time_filter(), limit=limit)
        return []

    def _load_seen_ids(self) -> set[str]:
        """grab previously processed post ids to avoid recollecting them."""
        if not self.seen_registry_path.exists():
            return set()
        try:
            with open(self.seen_registry_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                return {str(x) for x in data if x}
        except Exception as exc:
            logger.warning(f"unable to load seen ids registry ({exc})")
            return set()

    def _persist_seen_ids(self) -> None:
        """persist seen post ids to disk."""
        try:
            with open(self.seen_registry_path, "w", encoding="utf-8") as handle:
                json.dump(sorted(self._seen_ids), handle, indent=2)
        except Exception as exc:
            logger.warning(f"unable to persist seen ids registry ({exc})")

    def _check_for_image(self, post):
        """ check if post has an image or video (including galleries/direct image links)."""
        post_hint = getattr(post, "post_hint", None)
        if post_hint in {"image", "hosted:video", "rich:video"}:
            return True
        if getattr(post, "is_gallery", False):
            return True
        url_lower = (getattr(post, "url", "") or "").lower()
        if url_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".gifv")):
            return True

        return False
    # =======================================================================
    # fetch posts from one subreddit
    # =======================================================================
    def fetch_subreddit_posts(self, subreddit_name, sort="hot"):
        """
        fetch posts from a single subreddit using a specific sort method.
        return a DataFrame or None.
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = self._get_sort_stream(subreddit, sort, REDDIT_POSTS_PER_SUBREDDIT)
            out_rows = []

            for post in posts:
                # skip posts we've already processed in previous runs.
                if post.id in self._seen_ids:
                    continue

                # skip posts that are older than the cutoff date.
                created = datetime.fromtimestamp(post.created_utc)
                if created < self.cutoff_date:
                    continue

                # skip posts that are images or videos (including galleries/direct image links).
                if self._check_for_image(post):
                    continue

                # grab the flair (testing w/ WSB)
                flair_text = (post.link_flair_text or "").strip().lower()

                # build structured row.
                out_rows.append({
                    "created_utc": created,
                    "id": post.id,
                    "subreddit": subreddit_name,
                    "flair": flair_text,
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "title": post.title,
                    "text": post.selftext,
                    "link": f"https://www.reddit.com{post.permalink}",
                })

            return pd.DataFrame(out_rows) if out_rows else None

        except Exception as e:
            logger.error(f"error fetching r/{subreddit_name}: {e}")
            return None

    # =======================================================================
    # fetch all subreddits + save CSV
    # =======================================================================
    def fetch_all_subreddits(self, test_mode=False):
        print(f"{Fore.CYAN}===== stage 1: reddit data collection ====={Style.RESET_ALL}")
        print(f"lookback={self.max_days_lookback} days | limit={REDDIT_POSTS_PER_SUBREDDIT}\n")

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

                df = self.fetch_subreddit_posts(name, sort=sort)

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

        """
        # update seen-id registry with everything just collected.
        if "id" in final.columns and not final.empty:
            new_ids = {str(i) for i in final["id"].dropna().astype(str) if i}
            if new_ids:
                self._seen_ids.update(new_ids)
                self._persist_seen_ids()
        """
        
        filename = f"reddit_posts_{self.run_id}.csv"
        output_path = self.data_dir / filename
        final.to_csv(output_path, index=False)
        self.collected_data_path = output_path

        print(f"{Fore.GREEN}✓ collected {len(final)} total posts{Style.RESET_ALL}")
        print(f"  saved to: {output_path}\n")

        return output_path
