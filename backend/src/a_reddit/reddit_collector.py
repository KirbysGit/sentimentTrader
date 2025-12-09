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
      - _extract_post_media_info
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

# load environment vars
backend_dir = Path(__file__).parent.parent.parent
env_path = backend_dir / ".env"
load_dotenv(env_path if env_path.exists() else None)

# setup logging.
logger = logging.getLogger(__name__)


# ===========================================================================
# reddit data collector class
# ===========================================================================
class RedditDataCollector:

    def __init__(self, data_dir=None, max_days_lookback=30, run_date=None, run_id=None, enable_seen_registry: bool = True):
        self._run_ts = datetime.now(timezone.utc)                                                       # run timestamp.
        self.run_date = run_date or self._run_ts.date().isoformat()                                     # run date.
        self.run_id = run_id or self._run_ts.strftime("%Y%m%d_%H%M%S")                                  # run id.
        self.data_dir = data_dir or self._build_day_dir(RAW_REDDIT_DIR, self.run_date)                  # data directory for raw data output.
        self.max_days_lookback = max_days_lookback                                                      # max # of days to look back for data.
        
        os.makedirs(self.data_dir, exist_ok=True)                                                       # create data directory if it doesn't exist.

        self.enable_seen_registry = enable_seen_registry                                                # enable seen post registry.
        self.seen_registry_path = PROCESSED_REDDIT_BY_DAY_DIR.parent / "seen_post_ids.json"             # path to seen post ids registry.
        self._seen_ids = self._load_seen_ids() if self.enable_seen_registry else set()                  # load seen ids.

        # initialize PRAW client
        self.reddit = Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="script:MarketSentiment:v1 (u/kiiiiiiiiirb)",
            refresh_token=os.getenv("REFRESH_TOKEN"),
        )

        # cutoff timestamp for filtering old posts
        self.cutoff_date = datetime.now() - timedelta(days=max_days_lookback)
        self.last_output_path = None

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
        if not self.enable_seen_registry:
            return set()
        if not self.seen_registry_path.exists():
            return set()
        try:
            with open(self.seen_registry_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                return {str(x) for x in data if x}
        except Exception as exc:
            logger.warning(f"unable to load seen ids registry ({exc})")
            return set()

    # -----------------------
    # author metadata helper
    # -----------------------
    def _extract_author_info(self, post):
        """extract author metadata safely (some posts have no author)."""
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
    # url domain / media type helper
    # -----------------------------
    def _extract_post_media_info(self, post):
        """
        classify post type & domain.
        uses common reddit post_hint values when available.
        """
        url = post.url or ""
        domain = urlparse(url).netloc or None

        post_hint = getattr(post, "post_hint", None)

        is_self = post.is_self
        is_image = post_hint in ["image"]
        is_link = (not is_self) and post_hint in ["link", "rich:link"]
        is_video = post_hint in ["hosted:video", "rich:video"]

        # crosspost data (optional).
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
    # enhanced comment extraction
    # -----------------------------
    def _get_top_comments(self, post, max_count=5):
        """
        extract top comments with score & author info.
        return list of dicts.
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
    # fetch posts from one subreddit
    # =======================================================================
    def fetch_subreddit_posts(self, subreddit_name, limit=100, sort="hot"):
        """
        fetch posts from a single subreddit using a specific sort method.
        return a DataFrame or None.
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = self._get_sort_stream(subreddit, sort, limit)
            out_rows = []

            for post in posts:
                # skip posts we've already processed in previous runs
                if self.enable_seen_registry and post.id in self._seen_ids:
                    continue

                # skip posts that are older than the cutoff date.
                created = datetime.fromtimestamp(post.created_utc)
                if created < self.cutoff_date:
                    continue

                # author metadata.
                author_name, author_karma, author_is_mod, author_created = \
                    self._extract_author_info(post)

                # post media metadata.
                media_info = self._extract_post_media_info(post)

                # build structured row.
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

                    # author metadata.
                    "author_name": author_name,
                    "author_karma": author_karma,
                    "author_is_mod": author_is_mod,
                    "author_created_utc": author_created,

                    # media metadata.
                    **media_info,

                    # enhanced comments.
                    "top_comments": self._get_top_comments(post),
                })

            return pd.DataFrame(out_rows) if out_rows else None

        except Exception as e:
            logger.error(f"error fetching r/{subreddit_name}: {e}")
            return None

    # =======================================================================
    # fetch all subreddits + save CSV
    # =======================================================================
    def fetch_all_subreddits(self, limit=50, test_mode=False):
        print(f"{Fore.CYAN}===== stage 1: reddit data collection ====={Style.RESET_ALL}")
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

        filename = f"reddit_posts_{self.run_id}.csv"
        output_path = self.data_dir / filename
        final.to_csv(output_path, index=False)
        self.last_output_path = output_path

        print(f"{Fore.GREEN}✓ collected {len(final)} total posts{Style.RESET_ALL}")
        print(f"  saved to: {output_path}\n")

        return output_path
