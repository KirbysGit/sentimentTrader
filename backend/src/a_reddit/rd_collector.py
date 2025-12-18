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

# local imports.
from src.utils.config import (SUBREDDITS, SORT_METHODS, LOOKBACK, NUM_POSTS)
from src.utils.path_config import (env_path, seen_path, raw_reddit_dir)

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
        self.lookback = LOOKBACK
        self.num_posts = NUM_POSTS
        self.subreddits = SUBREDDITS
        self.sorts = SORT_METHODS

        # -- 4. get time filter
        self.time_filter = self.get_time_filter()

        # -- 5. set up our seen directory.
        self.seen = self.load_seen_ids(seen_path)

        # -- 6. build data directory based on day.
        self.data_dir = self.build_day_dir(raw_reddit_dir, self.run_date)
        os.makedirs(self.data_dir, exist_ok=True)

    # --- helper functions.

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

    def get_time_filter(self) -> str:
        # based on lookback value, adjust sort.
        if self.lookback < 1:
            return "hour"
        elif self.lookback == 1:
            return "day"
        elif self.lookback <= 7:
            return "week"
        elif self.lookback <= 30:
            return "month"
        else:
            return "year"    

    def load_seen_ids(self, seen_path) -> set[str]:
        if not seen_path.exists():                                  # verify path exists.
            return set()
        try:                                                        # if it does.
            with open(seen_path, "r", encoding="utf-8") as file:    # open file up.
                data = json.load(file)                              # parses json file into py objs.
                return {str(x) for x in data if x}                  # for piece of data, turn to string.
        except Exception as e:
            print("can't find seen ids 😭")
            return set()
    
    def add_seen_ids(self) -> None:
        try:
            with open(seen_path, "w", encoding="utf-8") as file:    # write new seen_ids to json.
                json.dump(sorted(self.seen), file, indent=2)
        except Exception as e:
            print(f"unable to add seen ids 😭")
    
    def check_for_image(self, post):
        post_hint = getattr(post, "post_hint", None)                                    # get post hint val.

        if post_hint in {"image", "hosted:video", "rich:video"}:                        # if post is image or vid.
            return True                                                                 # then, get rid of it.

        if getattr(post, "is_gallery", False):                                          # if post is gallery.
            return True                                                                 # then get rid of it.
        url_lower = (getattr(post, "url", "") or "").lower()
        if url_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".gifv")):     # if post ends with image.
            return True                                                                 # then get rid of it.

        return False                                                                    # else keep post.
    
    # --- main data fetching and processing ---

    def get_posts(self, subreddit, sort) -> []:
        if sort == "hot":
            return subreddit.hot(limit=self.num_posts)
        if sort == "new":
            return subreddit.new(limit=self.num_posts)
        if sort == "top":
            return subreddit.top(limit=self.num_posts, time_filter=self.time_filter)
        return []

    def fetch_subreddit_posts(self, name: str, sort: str):
        try:
            subreddit = self.reddit.subreddit(name)                     # get actual subreddit name.
            posts = self.get_posts(subreddit, sort)                     # fetch posts from subreddit w/ sort.
            clean = []

            for post in posts:                                          # iterate through posts.
                if post.id in self.seen:                                # if we've already seen post.
                    continue                                            # skip it.

                if self.check_for_image(post):                          # if post has image.
                    continue                                            # skip it.

                body = post.selftext if post.selftext else ""
                if self.is_unrenderable_text(body):                     # if unsupported text format.
                    continue
                
                flair = (post.link_flair_text or "").strip().lower()    # grab post flair.

                clean.append({                                          # set up dict per post data.
                    "created_utc": post.created_utc,
                    "id": post.id,
                    "subreddit": name,
                    "flair": flair,
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "title": post.title,
                    "text": post.selftext,
                    "link": f"https://www.reddit.com{post.permalink}",
                })

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
        
        pre_clean = len(pd.concat(all_dfs))                                         # grab pre-clean len.

        final = (                                                                   
            pd.concat(all_dfs, ignore_index=True)                                   # concatenate all dfs.
            .drop_duplicates(subset=["id"])                                         # drop any duplicates.
            .sort_values("created_utc", ascending=False)                            # sort by created time.
        )

        if not final.empty:                                                         # if final isn't empty.
            new_ids = {str(i) for i in final["id"].dropna().astype(str) if i}       # grab list of new ids.
            if new_ids:
                self.seen.update(new_ids)                                           # update our local ids.
                self.add_seen_ids()                                                 # add those ids to the json.
        
        filename = f"reddit_posts_{self.run_id}.csv"                                # set up run name.
        output_path = self.data_dir / filename                                      # set up our output path.
        final.to_csv(output_path, index=False)                                      # df to csv.

        print(f"we got {Fore.GREEN}{len(final)}{Style.RESET_ALL} total posts from our original {Fore.YELLOW}{pre_clean}{Style.RESET_ALL} posts")
        
        return output_path