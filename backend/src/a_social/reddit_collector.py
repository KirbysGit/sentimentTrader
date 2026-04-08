# part A of the collection phase.

# collecting the data from reddit posts per subreddit.

# output found in : raw / reddit / YYYY / MM / DD / reddit_posts_<run_id>.csv

# imports.
import os
import json
import pandas as pd
from praw import Reddit
from pathlib import Path
from dotenv import load_dotenv
from colorama import Fore, Style
from datetime import datetime, timezone, timedelta

# local imports.
from src.utils.config import (subreddits, sort_methods, lookback, num_posts, max_comments_per_post, max_comments_chars)
from src.utils.path_config import (env_path, last_seen_created_utc_path, raw_reddit_dir)

# get our .env vars.
load_dotenv(env_path)

class RedditCollector:

    # --- self-initialize.
    def __init__(self, run_date: str, run_id: str):
        
        # --- 1. set the run id.
        self.run_id = run_id
        self.run_date = run_date
        
        # --- 2. initialize how our reddit api.
        self.reddit = Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="script:MarketSentiment:v1 (u/kiiiiiiiiirb)",
            refresh_token=os.getenv("REFRESH_TOKEN"),
        )

        # -- 3. set up some pipeline config data.
        self.lookback = lookback
        self.num_posts = num_posts
        self.subreddits = subreddits
        self.sorts = sort_methods

        # -- 4. cursor for incremental "newer-than-last-run" fetching.
        self.last_seen_created_utc = self.load_cursor(last_seen_created_utc_path)
        self.cutoff_utc = (datetime.now(timezone.utc) - timedelta(days=int(self.lookback))).timestamp()

        # -- 6. build data directory based on day.
        self.data_dir = self.build_day_dir(raw_reddit_dir, self.run_date)
        os.makedirs(self.data_dir, exist_ok=True)

    # --- helper functions.

    @staticmethod
    # grab the time for the last seen post in each subreddit.
    def load_cursor(path: Path) -> dict[str, float]:
        # file is just {subreddit: last_seen_created_utc}
        if not path.exists():
            return {}
        try:
            # open the json file and load the data.
            data = json.loads(path.read_text(encoding="utf-8") or "{}")

            # iterate through the data and convert to float.
            out: dict[str, float] = {}
            for k, v in data.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue

            # return the data.
            return out
        except Exception:
            return {}

    @staticmethod
    # save time for last seen post in each subreddit.
    def save_cursor(path: Path, cursor: dict[str, float]) -> None:
        try:
            path.write_text(json.dumps(cursor, indent=2), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    # build the day directory from our current date.
    def build_day_dir(root: Path, run_date: str) -> Path:
        try:
            year, month, day = run_date.split("-")
        except Exception as e:
            print(f"issue w/ day dir : {e}")

        return root / year / month / day

    @staticmethod
    # check if the text is unrenderable.
    def is_unrenderable_text(text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        markers = (
            "not supported on old reddit",
            "click here to view the full post",
        )
        return any(marker in lower for marker in markers)

    @staticmethod
    # clean the comment's attached text.
    def clean_comment_text(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        low = s.lower()
        if low in {"[deleted]", "[removed]"}:
            return ""
        return " ".join(s.split())

    def get_comments_text(self, post) -> str:
        # grabs top N comments and concatenate bodies.
        try:
            post.comment_sort = "top"
            post.comments.replace_more(limit=0)

            # iterate through the comments and clean the text.
            parts = []
            
            for c in post.comments[: int(max_comments_per_post)]:
                body = self.clean_comment_text(getattr(c, "body", "") or "")
                if body:
                    parts.append(body)
                if sum(len(x) for x in parts) >= int(max_comments_chars):
                    break

            # join the parts and strip whitespace.
            out = "\n".join(parts).strip()
            return out[: int(max_comments_chars)]
        except Exception:
            return ""
    
    # --- main data fetching and processing ---

    def fetch_subreddit_posts(self, name: str, sort: str):
        try:
            # -- 1. get subreddit name.
            subreddit = self.reddit.subreddit(name)

            # -- 2. grab new posts.
            posts = subreddit.new(limit=self.num_posts)

            # -- 3. initialize our clean list.
            clean = []

            # -- 4. get the last seen post time for the subreddit.
            since_utc = float(self.last_seen_created_utc.get(name, 0.0) or 0.0)
            max_created_utc = since_utc

            # -- 5. iterate through posts.
            for post in posts:
                # -- 5.1. get the post creation time.
                created_utc = float(getattr(post, "created_utc", 0.0) or 0.0)

                # -- 5.2. if post was posted before our last seen post time, break.
                if created_utc and created_utc < self.cutoff_utc:
                    break
                if created_utc and created_utc <= since_utc:
                    break

                # -- 5.3. get the post body.
                #    NOTE: we keep media/link posts; title-only still carries tickers/sentiment.
                body = post.selftext if post.selftext else ""

                # -- 5.4. if unsupported text format, skip.
                if self.is_unrenderable_text(body):
                    continue
                
                # -- 5.5. grab post flair.
                flair = (post.link_flair_text or "").strip().lower()

                # -- 5.6. get the comments text.
                comments_text = self.get_comments_text(post)

                # -- 5.7. set up dict per post data.
                clean.append({
                    "created_at": datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat(),
                    "id": post.id,
                    "subreddit": name,
                    "flair": flair,
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "title": post.title,
                    "text": post.selftext,
                    "comments_text": comments_text,
                    "link": f"https://www.reddit.com{post.permalink}",
                })

                # -- 5.8. update the max created utc.
                if created_utc and created_utc > max_created_utc:
                    max_created_utc = created_utc

            # -- 5.9. update the cursor for next run.
            if max_created_utc > since_utc:
                self.last_seen_created_utc[name] = max_created_utc

            return pd.DataFrame(clean) if clean else None
        
        except Exception as e:
            print(f"issue w/ r/{name}: {e}")
            return None

    def fetch_data(self):

        print(f"{Fore.CYAN}=== stage 1 : reddit data collection ==={Style.RESET_ALL}")
        print(f"lookback = {self.lookback} days")
        print(f"attempting to fetch {self.num_posts} posts per sort\n")

        step = 0
        total = len(self.subreddits) * len(self.sorts)
        all_dfs = []

        # -- 1. iterate through all our config subreddits.
        for name in self.subreddits:
            print(f"{Fore.YELLOW}r/{name}:{Style.RESET_ALL}")
            # -- 1.1. iterate through our sorts.
            for sort in self.sorts:
                step += 1
                print(f"  [{step}/{total}] {sort}... ", end="", flush=True)
                
                # -- 1.2. fetch posts per sort.
                df = self.fetch_subreddit_posts(name, sort)

                # -- 1.3. if dataframe exists, add to our parent array.
                if df is not None and not df.empty:
                    all_dfs.append(df)
                    print(f"{Fore.GREEN}we got {len(df)} posts {Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}we got no posts from r/{name}{Style.RESET_ALL}")
            print()

        if not all_dfs:
            print(f"{Fore.RED}we got no posts from any subreddits{Style.RESET_ALL}")
            return None

        # -- 2. grab pre-clean len.
        pre_clean = len(pd.concat(all_dfs))

        # -- 3. concatenate all dfs, drop duplicates, and sort by created time.
        final = (
            pd.concat(all_dfs, ignore_index=True)
            .drop_duplicates(subset=["id"])
            .sort_values("created_at", ascending=False)
        )

        # -- 4. persist cursor (so next run is "newer chronological posts only").
        self.save_cursor(last_seen_created_utc_path, self.last_seen_created_utc)
        
        # -- 5. set up run name and output path.
        filename = f"reddit_posts_{self.run_id}.csv"
        output_path = self.data_dir / filename

        # -- 6. create the directory if it doesn't exist.
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        final.to_csv(output_path, index=False)

        print(f"we got {Fore.GREEN}{len(final)}{Style.RESET_ALL} total posts from our original {Fore.YELLOW}{pre_clean}{Style.RESET_ALL} posts")
        
        return final

    # --- refresh window support ---
    def refresh_recent_posts(self, days: int = 7) -> pd.DataFrame:
        """
        Re-fetch a rolling window of recent posts by post_id so we can update score/num_comments
        (and therefore engagement) as posts become more popular over time.

        Source of truth for which post_ids to refresh is the processed master table `posts_all.csv`.
        This does NOT touch the incremental cursor (last_seen_created_utc).
        """
        # -- 1. get the path to the processed reddit by day directory.
        try:
            from src.utils.path_config import processed_reddit_by_day_dir
        except Exception:
            return pd.DataFrame()

        # -- 2. get the path to the master posts file.
        master_posts_path = processed_reddit_by_day_dir / "posts_all.csv"
        if not master_posts_path.exists():
            return pd.DataFrame()

        # -- 3. read the master posts file.
        try:
            old = pd.read_csv(master_posts_path)
        except Exception:
            return pd.DataFrame()

        if old is None or old.empty:
            return pd.DataFrame()

        # -- 4. identify last N days by created_at (UTC).
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
        created = pd.to_datetime(old.get("created_at", None), utc=True, errors="coerce")
        old = old.assign(_created_at=created)
        old = old[old["_created_at"].notna() & (old["_created_at"] >= cutoff)]
        if old.empty:
            return pd.DataFrame()

        post_ids = old.get("post_id", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist()
        if not post_ids:
            return pd.DataFrame()

        print(f"{Fore.CYAN}stage 1 - refreshing engagement for last {days} days ({len(post_ids)} posts){Style.RESET_ALL}")

        refreshed = []
        for pid in post_ids:
            try:
                post = self.reddit.submission(id=str(pid))
                created_utc = float(getattr(post, "created_utc", 0.0) or 0.0)
                refreshed.append(
                    {
                        "created_at": datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat() if created_utc else "",
                        "id": post.id,
                        "subreddit": str(getattr(post, "subreddit", "") or ""),
                        "flair": (getattr(post, "link_flair_text", "") or "").strip().lower(),
                        "score": getattr(post, "score", 0),
                        "upvote_ratio": getattr(post, "upvote_ratio", 0.0),
                        "num_comments": getattr(post, "num_comments", 0),
                        "title": getattr(post, "title", "") or "",
                        "text": getattr(post, "selftext", "") or "",
                        # comments_text intentionally omitted on refresh (expensive + not needed for aggregates)
                        "comments_text": "",
                        "link": f"https://www.reddit.com{getattr(post, 'permalink', '')}",
                    }
                )
            except Exception:
                continue

        df = pd.DataFrame(refreshed)
        if df.empty:
            return df

        # Deduplicate by id (keep last).
        df = df.drop_duplicates(subset=["id"], keep="last")
        return df