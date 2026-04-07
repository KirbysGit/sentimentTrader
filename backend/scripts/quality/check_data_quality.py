"""quick script to check data quality and coverage after runs (run from repo any cwd: python scripts/quality/check_data_quality.py from backend)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

# backend/ is two levels above this file: scripts/quality -> scripts -> backend
BACKEND_ROOT = Path(__file__).resolve().parents[2]
base = BACKEND_ROOT / "data" / "processed"

# reddit daily
reddit_path = base / "reddit" / "by_day" / "reddit_daily_all.csv"
if reddit_path.exists():
    df_r = pd.read_csv(reddit_path)
    print(f"\n=== REDDIT DAILY ===")
    print(f"Rows: {len(df_r):,}")
    print(f"Unique dates: {df_r['date'].nunique()} days")
    print(f"Date range: {df_r['date'].min()} to {df_r['date'].max()}")
    print(f"Unique tickers: {df_r['ticker'].nunique()}")
    print(f"Avg rows per day: {len(df_r) / df_r['date'].nunique():.1f}")
    print(f"Top 5 tickers by mention_count: {df_r.groupby('ticker')['mention_count'].sum().nlargest(5).to_dict()}")

    d_r = pd.to_datetime(df_r["date"], errors="coerce").dropna()
    if not d_r.empty:
        rmax = d_r.max().date()
        today = date.today()
        stale = (today - rmax).days
        print(f"Latest reddit_daily date: {rmax} ({stale} calendar day(s) behind today)")
        if stale > 2:
            print(f"  WARNING: pipeline may have missed runs — investigate.")

        dr = pd.date_range(d_r.min().normalize(), d_r.max().normalize(), freq="D")
        have = set(d_r.dt.normalize())
        gaps = [d.date().isoformat() for d in dr if d not in have]
        print(f"Calendar days in range with zero reddit_daily rows: {len(gaps)}")
        if gaps and len(gaps) <= 20:
            print(f"  Missing dates: {', '.join(gaps)}")
        elif gaps:
            print(f"  First 10 missing: {', '.join(gaps[:10])} ...")

    recent = df_r.groupby("date", sort=False).size().tail(7)
    print(f"Rows per date (last 7 present dates): {recent.to_dict()}")

# posts master
posts_path = base / "reddit" / "by_day" / "posts_all.csv"
if posts_path.exists():
    df_p = pd.read_csv(posts_path)
    df_p["date"] = pd.to_datetime(df_p["created_at"]).dt.date.astype(str)
    print(f"\n=== POSTS MASTER ===")
    print(f"Rows: {len(df_p):,}")
    print(f"Date range: {df_p['date'].min()} to {df_p['date'].max()}")
    print(f"Unique post_ids: {df_p['post_id'].nunique()}")
    print(f"Avg engagement: {df_p['engagement'].mean():.1f}")

# google trends
trends_path = base / "trends" / "by_day" / "google_trends_daily_all.csv"
if trends_path.exists():
    df_t = pd.read_csv(trends_path)
    print(f"\n=== GOOGLE TRENDS ===")
    print(f"Rows: {len(df_t):,}")
    print(f"Unique dates: {df_t['date'].nunique()} days")
    print(f"Date range: {df_t['date'].min()} to {df_t['date'].max()}")
    print(f"Unique tickers: {df_t['ticker'].nunique()}")
    missing_trends = df_t["trends_interest"].isna().sum()
    print(f"Missing trends_interest: {missing_trends} rows ({missing_trends / len(df_t) * 100:.1f}%)")

# stock labels
stocks_path = base / "stocks" / "by_day" / "stock_labels_all.csv"
df_s = None
if stocks_path.exists():
    df_s = pd.read_csv(stocks_path)
    print(f"\n=== STOCK LABELS ===")
    print(f"Rows: {len(df_s):,}")
    print(f"Unique dates: {df_s['date'].nunique()} days")
    print(f"Date range: {df_s['date'].min()} to {df_s['date'].max()}")
    print(f"Unique tickers: {df_s['ticker'].nunique()}")
    has_labels = df_s["y_ret_1d"].notna().sum()
    print(f"Rows with y_ret_1d labels: {has_labels:,} ({has_labels / len(df_s) * 100:.1f}%)")

# merged training features
merged_path = base / "metrics" / "merged_features_all.csv"
if merged_path.exists():
    df_m = pd.read_csv(merged_path)
    print(f"\n=== MERGED FEATURES (training) ===")
    print(f"Rows: {len(df_m):,}")
    if "date" in df_m.columns:
        print(f"Date range: {df_m['date'].min()} to {df_m['date'].max()}")
    if "ticker" in df_m.columns:
        print(f"Unique tickers: {df_m['ticker'].nunique()}")
    if "y_ret_1d" in df_m.columns:
        na = df_m["y_ret_1d"].isna().sum()
        print(f"Rows with null y_ret_1d: {na} ({100 * na / len(df_m):.1f}%)")

# overlap check
if reddit_path.exists() and stocks_path.exists() and df_s is not None:
    print(f"\n=== OVERLAP CHECK ===")
    df_r = pd.read_csv(reddit_path)
    df_s = pd.read_csv(stocks_path)
    reddit_dates = set(df_r["date"].unique())
    stock_dates = set(df_s["date"].unique())
    overlap_dates = reddit_dates & stock_dates
    print(f"Reddit dates: {len(reddit_dates)}, Stock dates: {len(stock_dates)}")
    print(f"Overlapping dates: {len(overlap_dates)}")

    reddit_tickers = set(df_r["ticker"].unique())
    stock_tickers = set(df_s["ticker"].unique())
    overlap_tickers = reddit_tickers & stock_tickers
    print(f"Reddit tickers: {len(reddit_tickers)}, Stock tickers: {len(stock_tickers)}")
    print(f"Overlapping tickers: {len(overlap_tickers)}")

    merged = df_r.merge(df_s, on=["date", "ticker"], how="inner", suffixes=("_r", "_s"))
    print(f"Potential training rows (date+ticker overlap): {len(merged):,}")
    if len(merged) > 0:
        has_label = merged["y_ret_1d"].notna().sum()
        print(f"  Rows with y_ret_1d labels: {has_label:,} ({has_label / len(merged) * 100:.1f}%)")

print("\n=== RECOMMENDATION ===")
if reddit_path.exists() and stocks_path.exists():
    df_r = pd.read_csv(reddit_path)
    df_s = pd.read_csv(stocks_path)
    merged = df_r.merge(df_s, on=["date", "ticker"], how="inner", suffixes=("_r", "_s"))
    trainable = merged[merged["y_ret_1d"].notna()]
    if len(trainable) < 100:
        print("WARNING: Less than 100 trainable rows. Consider backfilling historical Reddit data.")
    elif len(trainable) < 500:
        print("WARNING: Less than 500 trainable rows. May want to backfill for better model training.")
    else:
        print(f"OK: {len(trainable):,} trainable rows - sufficient for initial model training.")
        print("  No backfill needed unless you want more historical context.")
