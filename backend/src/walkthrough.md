# pipeline walkthrough (by folder)

## `orchestrator.py` - main pipeline file

this file is used for referencing and connecting all of our separate processes i'll review below.

---

## step 0 - `orchestrator.py` - initialization of pipeline & relevant data

---

pipeline starts off with the initialization call of the pipeline object.

creates three pieces of data for us :

    - run_ts    ->      runtime of initial pipeline call (utc timezone)
    - run_date  ->      date of initial pipeline call
    - run_id    ->      individual id of run based on runtime

---

## step 1 - `a_social` - social media collection

---

#### step 1.1 - `__init__` - initialize reddit collection data.

---

first off we accept the `run_date` and `run_id` from `orchestrator.py`.

we then initialize our reddit praw api w/ our client id & secrets.

initialize some of our pipeline config data including :

    - lookback      ->      how far the praw api should look back for posts (e.g. 1 day, 1 week, 1 month)
    - num_posts     ->      # of posts to grab per sort per subreddit.
    - subreddits    ->      subreddits for the api to parse through (e.g. r/wallstreetbets, r/finance)
    - sorts         ->      sorts to reference per subreddit (e.g. hot, new, top)

then some of our helper data :

    - last_seen_created_utc     ->      tells us when the latest post was grabbed 
    - cutoff                    ->      earliest timestamp for posts to be collected (now - lookback)

then build our directories to save our collected data based on our run ids :
    
    - raw / reddit / year / month / day / reddit_posts_YYYYMMDD_hhmmss

---
#### step 1.2 - `fetch_reddit_data` - collect relevant reddit data.
---

now to begin the collection :

-   we iterate through all of the config subreddits (`self.subreddits`)
    -   per subreddit we iterate through all of the sorts (`self.sorts`)
        -   per sort we fetch the posts per sort (`fetch_subreddit_posts(name, sort)`)
            -   per fetch we ->
                -   grab post data w/ proper handling of empty or edge cases.
                -   verify it exists after our cutoff date
                -   update our `last_seen_created_utc` if applicable
                -   return the data like this :
                        
                        {
                            created_at      ->  datetime of post creaetion
                            id              ->  id of post based on api response
                            subreddit       ->  name of subreddit we pulled from
                            flair           ->  name of flair (e.g discussion, news, meme)
                            score           ->  score of the posts (upvotes)
                            upvote_ratio    ->  ratio of upvotes to downvotes
                            num_comments    ->  # of comments
                            title           ->  title of the post
                            text            ->  main body text of post
                            comments_text   ->  return top (max_comments_per_post) comments text
                            link            ->  link to the relevant post
                        }

                - append the data to our dataframe
                
- clean the df, dropping duplicates, sorting by the created time of post.

- save df to our designated file in the directory.

- return csv output path.

---

#### step 1.3 - `refresh_recent_posts` - grab fresh reddit metadata.

---

this is a second call on the reddit collection data, where we :

-   check over all posts within the last amount of `days` (default 7).
-   collect those `post_ids`
-   per post add to a new df called `refreshed` w/ updated `score`, `upvote_ratio`, and `num_comments`

---

#### step 1.4 - `orchestrator.py` - compare engagement

---

iterate through our two dataframes `new` and `refreshed`

-   compare engagement per post (calculated between score & num_comments)
-   keep highest engagement per post id


---

## step 2 - `b_analysis` - analyze & breakdown our collected reddit data.

---

#### step 2.1 - `__init__` - initialize reddit processor data.

---

we initialize our reddit data processor in this order :

- set up etf & equity universes.

    ```
    etf_universe    ->      data of all etfs in our universe (based on etfs.csv)
    equity_universe ->      data of all equities in our universe (based on equities.csv)
    ```

- build name maps for tickers.

    ```
    ticker_by_name  ->      map of ticker name to ticker symbol. (e.g. "Apple Inc." -> "AAPL")
    aliases         ->      map of ticker aliases to ticker symbol. (e.g. "Facebook" -> "META")

- set up ticker debugging funnel (extra step for data purposes).

- set up ticker processing info.

    ```
    agg_scores      ->      dictionary of ticker scores.
    agg_counts      ->      dictionary of ticker counts.
    ```

- set up sentiment scorer.

    ```
    sentiment_scorer ->      instance of our sentiment scorer class.
    ```

- set up post-ticker records (for output).

    ```
    records     ->      list of records of posts & their tickers.
    posts       ->      list of posts with their basic metrics.

---

#### step 2.2 - `process` - process the reddit data.

---

we iterate through our dataframe from the previous collection and per row :

-   grab the title and text for extraction.
-   extract tickers using regex and universe checks. (`extract_tickers`)
-   check for company names. (`company_names`)
-   combine # of hits of tickers and company names.
-   clear out some basic words i've run into. (`stop_words`)
-   boost the tickers using our confidence booster. (`boost_tickers`)
-   clean the boosted tickers df based on the max score and absolute floor. (`clean_boosted`)

if no cleaned tickers, move to next row.

-   if cleaned tickers, add the post to our posts list then
-   add the tickers to our `agg_scores` and `agg_counts`.
-   debug the reasoning. (`debug_reasonings`)
-   score the sentiment per ticker (per context window). (`sentiment_scorer`)
-   add the record to our `records` list.

after all of this, we sort the `agg_scores` and `agg_counts` by score and count respectively.

then set up our outputs.

---

#### step 2.3 - `posts_df` - set up posts dataframe.

---

df -> per day per reddit per post

this is our most basic dataframe that we use for our reddit collection.

this represents the basic reddit posts data for that run that made it past our initial filters.

we set up our posts dataframe by creating the directory :

    processed_reddit_by_day_dir / f"posts_{run_id}.csv"

as well as to the master posts table in the directory :

    processed_reddit_by_day_dir / "posts_all.csv"

---

#### step 2.4 - `post_ticker_df` - set up post-ticker dataframe.

---

df -> per day per reddit per post per ticker

this is our more quantitative dataframe that we use for our ticker analysis.

we iterate through our `records` list and group by the `created_date`, `post_id`, and `ticker` and aggregate the data to get : 

    - created_at      ->  datetime of post creaetion
    - post_id         ->  id of post based on api response
    - ticker          ->  ticker symbol
    - subreddit       ->  name of subreddit we pulled from
    - engagement      ->  engagement of the post
    - boost_score     ->  boost score of the post
    - post_sentiment  ->  sentiment score of the post

basically sorts the data per day, per post, per ticker.

so one row in this df is the tickers activity for a specific reddit thread for that day.

this will be used for our daily ticker analysis.

---

#### step 2.5 - `daily_df` - set up daily dataframe.

---

df -> per day per ticker

we iterate through our `post_ticker_df` and group by the `created_date` and `ticker` and aggregate the data to get : 

    - created_date          ->  date of the post
    - ticker                ->  ticker symbol
    - mention_count         ->  # of mentions of the ticker
    - total_engagement      ->  total engagement of the ticker
    - avg_sentiment         ->  average sentiment of the ticker
    - boost_score_sum       ->  total boost score of the ticker
    - subreddit_diversity   ->  # of different subreddits the ticker was mentioned in
    - weighted_numer        ->  weighted numer of the ticker

this will be used for our weighted sentiment calculation.

we then calculate the weighted sentiment by using the formula : 

    weighted_sentiment = weighted_numer / total_engagement

then we drop the `weighted_numer` column after adding the weighted sentiment.

now, concatenate our daily df with the master daily table by date and ticker

we save this to the master daily table in the directory :

    processed_reddit_by_day_dir / "reddit_daily_all.csv"

---

#### step 2.6 - return the post-ticker dataframe.

---

in terms of output for this file, we just return the `post_ticker_df`.

however the entire processor outputs :

```
- output #1 -> `posts_df` into both the `posts_all.csv` and the `posts_{run_id}.csv`
- output #2 -> `post_ticker_df`
- output #3 -> `daily_df` into the `reddit_daily_all.csv`
```

---

#### step 2.7 - `grab_top_tickers` - grab the top tickers from our df.

---

we pass our `post_ticker_df` to this function to grab the top tickers based on our config.

this function :

-   group by the `ticker` and aggregate the data to get :
    
    ```
    - mention_count     ->  # of mentions of the ticker
    - total_engagement  ->  total engagement of the ticker
    - boost_score_sum   ->  total boost score of the ticker
    - weighted_numer    ->  weighted numer of the ticker
    ```
    
-   then we calculate the weighted sentiment by using the formula : 

    ```
    weighted_sentiment = weighted_numer / total_engagement
    ```

-   then we keep everything with at least 1 mention. (can be changed)

-   then we calculate the trend strength by using the formula : 

    ```
    trend_strength = boost_score_sum * (total_engagement + 1) ** 0.5
    ```

-   then we sort the data by trend strength and take the top `topN` tickers.

-   then we save the watchlist to the directory :

    ```
    processed_reddit_by_day_dir / f"watchlist_{run_id}.csv"
    ```

-   then we return the top `topN` tickers.

---

## step 3 - `c_features` - source features from stocktwits (not used) and google trends.

---

#### step 3.1 - `__init__` - initialize google trends collector data.

----

we initialize our google trends collector in this order :

```
-   geo             ->  the geo to collect data from (default US)
-   timeframe       ->  the timeframe to collect data from (default now 7-d)
-   max_retries     ->  the max number of retries to collect data (default 5)
-   base_delay_s    ->  the base delay in seconds between retries (default 3.0)
```

then set up the pytrends api with some specific settings to reduce rate limiting and bot detection.

---

#### step 3.2 - `collect` - collect the google trends data.

---

we take our `tickers` list and pass it to this function to collect the google trends data.

it works in this order :

-   create the output directory and saves the daily data like this :
    
    ```
    directory : processed_trends_by_day_dir
    daily file : f"google_trends_daily_{run_id}.csv"
    master file : processed_trends_by_day_dir / "google_trends_daily_all.csv"
    ```

- build the terms for the pytrends api. (`build_terms`)

- initialize the payload for the pytrends api.

- iterates through the pairs and builds the payload by batching the requests.

- iterate through the returned batch and add the data to the rows list.

- concat the rows into a single dataframe.

- calculate the day-over-day + rolling baseline features (per ticker) because :
```
for day-over-day -> when we grab the query normalized between 0 and 100, AAPL is 40 today might not mean anything by itself, so we want to see how it trends over time.

for 7d rolling baseline + z + ratio -> we want to see how the query is performing relative to the rolling baseline. basically asking, "is today weird vs this tickers last few days?"

we calculate these so the downstream features mean momentum or suprise rather than just raw levels.
```

- the data then looks like this :

```
date                  -> date of the data
ticker                -> ticker symbol
trends_interest       -> normalized interest of the query
trends_dod            -> day-over-day change in interest
trends_dod_pct        -> day-over-day change in interest as a percentage
trends_roll7_mean     -> 7d rolling mean of interest
trends_roll7_std      -> 7d rolling standard deviation of interest
trends_roll7_z        -> z-score of interest relative to the rolling baseline
trends_roll7_ratio    -> ratio of interest to the rolling baseline
```

- then save the dataframe to the daily file.

- update the master file and return the results.

---

## step 4 - `d_stocks` - collect stock data from yfinance.

---

#### step 4.1 - `collect_stock_data` - collect the stock data.

----

we take our `tickers` list and pass it to this function to collect the stock data.

it works in this order :

- create raw stock directory to hold our stock data (sorted by ticker).

- normalize the tickers and write the "ready-to-fetch" list.

- fetch the daily OHLCV. (open, high, low, close, volume) per ticker :

    - if we already have a file for this ticker, only fetch a small recent window and dedupe by date.

- save it to the raw stock directory (dependent on ticker) like this :
    
    ```
    directory : raw_stocks_dir / "by_ticker"
    file : f"raw_{ticker}.csv"
    ```

- the data then looks like this :

    ```
    date        ->  date of the data
    ticker      ->  ticker symbol
    open        ->  open price of the stock
    high        ->  high price of the stock
    low         ->  low price of the stock
    close       ->  close price of the stock
    volume      ->  volume of the stock
    adj_close   ->  adjusted close price of the stock
    ```

- then clean up the data and begin to build per-day, per-ticker labels / features for training introducing features like :

    ```
    - close_ret_3d  ->  3 day return of the stock
    - next_close    ->  next day close price of the stock
    - y_ret_1d      ->  1 day return of the stock
    ```
    
- creates the labels dataframe with this data.

- update our master table (`stock_labels_all.csv`) by appending the new data and deduping by date and ticker.

- then return the by_ticker_dir path to be used for the next step.

---

## step 5 - `e_merge` - merge the reddit data w/ the stock data.

---

### step 5.1 - `build_dataset` - builds our merged dataset.

---

we accept the following arguments :

    ```
    stocks_by_ticker_dir  ->  directory of the stock data.
    run_id                ->  id of the run.
    tickers               ->  list of tickers to merge.
    ```

it works in this order :
- set up directory for merged features.
- read the reddit daily all file.
- read the stock files and set up each row like this :
```
ticker          ->      ticker symbol
date            ->      date of the data
close           ->      close price of the stock
close_ret_3d    ->      3 day return of the stock
y_ret_1d        ->      1 day return of the stock
```
- concatenate the rows to a df.
- merge reddit daily with daily stock df, where we only keep the days where we have both sentiment + ohlcv.
- introduce some more features like `buzz`, `sentiment_chg_1d`, to our merged df.
- write the merged df to a csv with these columns :
```
ticker                          ->      ticker symbol
date                            ->      date of the data
weighted_sentiment              ->      weighted sentiment
buzz                            ->      log1p(total_engagement) (attention)
sentiment_chg_1d                ->      sentiment acceleration
weighted_sentiment_lag1         ->      weighted sentiment lag1
buzz_lag1                       ->      buzz lag1
buzz_dod                        ->      day-over-day change in buzz
weighted_sentiment_roll3_mean   ->      3 day rolling mean of weighted sentiment
weighted_sentiment_roll5_mean   ->      5 day rolling mean of weighted sentiment
had_reddit                      ->      1 if mention_count >= 1, 0 otherwise
```
- return the path to the merged features csv.