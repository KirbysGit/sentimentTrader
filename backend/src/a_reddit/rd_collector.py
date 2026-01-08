# first phase of pipeline.

# collecting the data from reddit posts per subreddit.

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
        self.last_seen_created_utc = load_cursor(last_seen_created_utc_path)
        self.cutoff_utc = (datetime.now(timezone.utc) - timedelta(days=int(self.lookback))).timestamp()

        # -- 6. build data directory based on day.
        self.data_dir = self.build_day_dir(raw_reddit_dir, self.run_date)
        os.makedirs(self.data_dir, exist_ok=True)

    # --- helper functions.

    def load_cursor(path: Path) -> dict[str, float]:
        # file is just {subreddit: last_seen_created_utc}
        if not path.exists():
            return {}
        try:
            # open the json file and load the data.
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
            if not isinstance(data, dict):
                return {}

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

    def save_cursor(path: Path, cursor: dict[str, float]) -> None:
        try:
            path.write_text(json.dumps(cursor, indent=2), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def build_day_dir(root: Path, run_date: str) -> Path:
        # get day dir from our current date.
        try:
            year, month, day = run_date.split("-")
        except Exception as e:
            print(f"issue w/ day dir : {e}")

        return root / year / month / day

    @staticmethod
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
            # get actual subreddit name.
            subreddit = self.reddit.subreddit(name)

            # we run incremental off "new" only.
            posts = subreddit.new(limit=self.num_posts)
            clean = []
            since_utc = float(self.last_seen_created_utc.get(name, 0.0) or 0.0)
            max_created_utc = since_utc

            for post in posts:                                          # iterate through posts.
                created_utc = float(getattr(post, "created_utc", 0.0) or 0.0)

                # chronological cutoffs: break early.
                if created_utc and created_utc < self.cutoff_utc:
                    break
                if created_utc and created_utc <= since_utc:
                    break

                # NOTE: we keep media/link posts; title-only still carries tickers/sentiment.
                body = post.selftext if post.selftext else ""

                # if unsupported text format, skip.
                if self.is_unrenderable_text(body):
                    continue
                
                # grab post flair.
                flair = (post.link_flair_text or "").strip().lower()

                # get the comments text.
                comments_text = self.get_comments_text(post)

                # set up dict per post data.
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

                # update the max created utc.
                if created_utc and created_utc > max_created_utc:
                    max_created_utc = created_utc

            # update the cursor for next run.
            if max_created_utc > since_utc:
                self.last_seen_created_utc[name] = max_created_utc

            return pd.DataFrame(clean) if clean else None               # return dataframe of posts.
        
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

        for name in self.subreddits:                                                # iterate through all our config subreddits.
            print(f"{Fore.YELLOW}r/{name}:{Style.RESET_ALL}")
            for sort in self.sorts:                                                 # per subreddit iterate through our sorts.
                step += 1
                print(f"  [{step}/{total}] {sort}... ", end="", flush=True)
                
                df = self.fetch_subreddit_posts(name, sort)                         # fetch posts per sort.

                if df is not None and not df.empty:                                 # if dataframe exists.
                    all_dfs.append(df)                                              # add dataframe to our parent array.
                    print(f"{Fore.GREEN}we got {len(df)} posts 🎉{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}we got no posts 😡{Style.RESET_ALL}")
            print()
        
        if not all_dfs:
            print(f"we got {Fore.GREEN}0{Style.RESET_ALL} total posts from our original {Fore.YELLOW}0{Style.RESET_ALL} posts")
            return None

        pre_clean = len(pd.concat(all_dfs))                                         # grab pre-clean len.

        final = (                                                                   
            pd.concat(all_dfs, ignore_index=True)                                   # concatenate all dfs.
            .drop_duplicates(subset=["id"])                                         # drop any duplicates.
            .sort_values("created_at", ascending=False)                             # sort by created time.
        )

        # persist cursor (so next run is "newer chronological posts only").
        self.save_cursor(last_seen_created_utc_path, self.last_seen_created_utc)
        
        filename = f"reddit_posts_{self.run_id}.csv"                                # set up run name.
        output_path = self.data_dir / filename                                      # set up our output path.
        final.to_csv(output_path, index=False)                                      # df to csv.

        print(f"we got {Fore.GREEN}{len(final)}{Style.RESET_ALL} total posts from our original {Fore.YELLOW}{pre_clean}{Style.RESET_ALL} posts")
        
        return output_path