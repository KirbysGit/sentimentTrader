"""
compare dumb baselines vs our logistic model on the exact same train/test split as train_baseline.

run from backend folder:
  python scripts/eval/eval_baselines.py
  python scripts/eval/eval_baselines.py --no-log
  python scripts/eval/eval_baselines.py --log path/to/custom_history.csv

each run appends one row to data/processed/metrics/eval_history.csv (simple numbers to compare over time).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from src.f_train.train_baseline import (  # noqa: E402
    BASELINE_FEATURE_COLS,
    active_feature_cols,
    prepare_merged_for_training,
    time_split,
)
from src.utils.path_config import processed_metrics_dir  # noqa: E402
from src.utils.config import logistic_l2_c, train_lookback_days  # noqa: E402


DEFAULT_LOG = processed_metrics_dir / "eval_history.csv"

# reddit-side columns vs price column (must stay in sync with feature_builder + train_baseline list).
PRICE_FEATURE_COLS = ["close_ret_3d"]
SENTIMENT_FEATURE_COLS = ["weighted_sentiment", "buzz", "sentiment_chg_1d"]


def _metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # score predictions like train_baseline: accuracy, precision, recall on up vs not up.
    return {
        "method": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _acc_by_name(summary: pd.DataFrame, substr: str) -> float:
    row = summary[summary["method"].str.contains(substr, regex=False)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["accuracy"])


def _append_log(log_path: Path, row: dict) -> None:
    # one csv with a row per eval run so you can scroll and compare without noise.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if log_path.exists():
        old = pd.read_csv(log_path)
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(log_path, index=False)


def _logreg():
    return LogisticRegression(max_iter=4000, C=float(logistic_l2_c), penalty="l2", solver="lbfgs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged",
        type=Path,
        default=processed_metrics_dir / "merged_features_all.csv",
        help="path to merged_features csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="optional: save the methods table for this run only",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="test fraction (same contract as train_baseline)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"append one summary row to this csv (default: {DEFAULT_LOG.name}). use --no-log to skip.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="do not append eval_history",
    )
    args = parser.parse_args()

    if not args.merged.exists():
        print(f"{Fore.RED}file not found: {args.merged}{Style.RESET_ALL}")
        sys.exit(1)

    df = prepare_merged_for_training(args.merged)
    if len(df) < 30:
        print(f"{Fore.RED}not enough rows after cleaning ({len(df)}). need more data.{Style.RESET_ALL}")
        sys.exit(1)

    train_df, test_df = time_split(df, test_frac=args.test_frac)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    y_train = train_df["y_up_1d"].astype(int).values
    y_test = test_df["y_up_1d"].astype(int).values

    rows_out = []

    pred = np.ones(len(y_test), dtype=int)
    rows_out.append(_metrics_row("always_up", y_test, pred))

    pred = np.zeros(len(y_test), dtype=int)
    rows_out.append(_metrics_row("always_down", y_test, pred))

    maj = 1 if y_train.mean() >= 0.5 else 0
    pred = np.full(len(y_test), maj, dtype=int)
    rows_out.append(_metrics_row(f"majority_train (vote {maj})", y_test, pred))

    mom = (test_df["close_ret_3d"].astype(float) > 0).astype(int).values
    rows_out.append(_metrics_row("momentum_sign (close_ret_3d > 0)", y_test, mom))

    # full model: same features as production; drop train-constant cols (e.g. had_reddit all 1).
    feature_fit = active_feature_cols(train_df)
    model = _logreg()
    model.fit(train_df[feature_fit], train_df["y_up_1d"])
    prob_up = model.predict_proba(test_df[feature_fit])[:, 1]
    pred_lr = (prob_up >= 0.5).astype(int)
    rows_out.append(_metrics_row("logistic (our model, prob >= 0.5)", y_test, pred_lr))

    # ablation: only 3-day return — strips out reddit/sentiment bundle so we see if accuracy barely moves.
    model_price = _logreg()
    model_price.fit(train_df[PRICE_FEATURE_COLS], train_df["y_up_1d"])
    pred_price = (model_price.predict_proba(test_df[PRICE_FEATURE_COLS])[:, 1] >= 0.5).astype(int)
    rows_out.append(_metrics_row("logistic ablation: price only (close_ret_3d)", y_test, pred_price))

    # ablation: sentiment + buzz + chg without price — see if anything is there without momentum feature.
    model_sent = _logreg()
    model_sent.fit(train_df[SENTIMENT_FEATURE_COLS], train_df["y_up_1d"])
    pred_sent = (model_sent.predict_proba(test_df[SENTIMENT_FEATURE_COLS])[:, 1] >= 0.5).astype(int)
    rows_out.append(_metrics_row("logistic ablation: sentiment only (3 reddit cols)", y_test, pred_sent))

    print(f"\n{Fore.CYAN}=== same split as train_baseline ({args.test_frac:.0%} test tail) ==={Style.RESET_ALL}")
    print(f"rows: train={len(train_df):,} test={len(test_df):,} | train_lookback_days={train_lookback_days}")
    print(f"test date range: {test_df['date'].min()} .. {test_df['date'].max()}\n")

    summary = pd.DataFrame(rows_out)
    print(summary.to_string(index=False))

    # quick read on which direction each feature pushes p(up) after fitting on train.
    print(f"\n{Fore.CYAN}=== logistic coefficients (full model on train) ==={Style.RESET_ALL}")
    print(f"features used ({len(feature_fit)}): {feature_fit}")
    print(f"intercept (log-odds bias): {float(model.intercept_[0]):.6f}")
    for name, coef in zip(feature_fit, model.coef_.ravel()):
        print(f"  {name}: {float(coef):.6f}")
    acc_full = float(accuracy_score(y_test, pred_lr))
    acc_price_only = float(accuracy_score(y_test, pred_price))
    acc_sent_only = float(accuracy_score(y_test, pred_sent))
    print(f"\n{Fore.CYAN}=== ablation readout (test accuracy) ==={Style.RESET_ALL}")
    print(f"full model:              {acc_full:.6f}")
    print(f"price only:              {acc_price_only:.6f}")
    print(f"sentiment only (3 cols): {acc_sent_only:.6f}")
    d_red = acc_full - acc_price_only
    d_px = acc_full - acc_sent_only
    print(f"full minus price_only:     {d_red:+.6f}  (positive => reddit cols help vs price-only on this slice)")
    print(f"full minus sentiment_only: {d_px:+.6f}  (positive => adding close_ret_3d helps vs sentiment-only)")

    s = pd.Series(prob_up)
    print(f"\n{Fore.CYAN}=== probability spread (logistic, test set only) ==={Style.RESET_ALL}")
    print(f"mean={s.mean():.4f} std={s.std():.4f} min={s.min():.4f} max={s.max():.4f}")
    low = (s < 0.45).mean()
    mid = ((s >= 0.45) & (s <= 0.55)).mean()
    high = (s > 0.55).mean()
    print(f"share prob < 0.45: {low:.1%}")
    print(f"share 0.45 <= prob <= 0.55: {mid:.1%}")
    print(f"share prob > 0.55: {high:.1%}")

    cal_spread = None
    try:
        rk = s.rank(method="first")
        bins = pd.qcut(rk, q=10, duplicates="drop")
        cal = test_df.assign(_p=s, _bin=bins).groupby("_bin", observed=True).agg(
            n=("_p", "count"),
            mean_prob=("_p", "mean"),
            frac_up=("y_up_1d", "mean"),
        )
        print(f"\n{Fore.CYAN}=== decile calibration (by rank, mean prob vs actual up rate) ==={Style.RESET_ALL}")
        print(cal.to_string())
        if len(cal) >= 2:
            cal_spread = float(cal["frac_up"].iloc[-1] - cal["frac_up"].iloc[0])
    except Exception as e:
        print(f"{Fore.YELLOW}could not build decile bins ({e}). skip.{Style.RESET_ALL}")

    yret = test_df["y_ret_1d"].astype(float)
    rho = None
    if s.nunique() > 1 and yret.nunique() > 1:
        rho = float(s.corr(yret, method="spearman"))
        print(f"\nspearman(prob_up, y_ret_1d) on test: {rho:.4f} (0 means no rank link)")
    else:
        print("\nspearman skipped: prob or returns are almost constant on this test slice")

    print()

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out_csv, index=False)
        print(f"wrote summary: {args.out_csv}")

    acc_lr = _acc_by_name(summary, "logistic")
    acc_maj = _acc_by_name(summary, "majority_train")
    acc_mom = _acc_by_name(summary, "momentum_sign")

    if not args.no_log:
        log_row = {
            "utc_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "merged_file": str(args.merged.name),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "test_frac": args.test_frac,
            "acc_logistic": round(acc_lr, 6),
            "acc_majority": round(acc_maj, 6),
            "acc_momentum": round(acc_mom, 6),
            "acc_minus_majority": round(acc_lr - acc_maj, 6),
            "acc_minus_momentum": round(acc_lr - acc_mom, 6),
            "acc_price_only": round(acc_price_only, 6),
            "acc_sentiment_only": round(acc_sent_only, 6),
            "acc_full_minus_price_only": round(acc_full - acc_price_only, 6),
            "acc_full_minus_sentiment_only": round(acc_full - acc_sent_only, 6),
            "prob_std": round(float(s.std()), 6),
            "prob_mid_share": round(float(mid), 6),
            "spearman_prob_yret": "" if rho is None else round(rho, 6),
            "cal_top_minus_bottom_frac_up": ""
            if cal_spread is None
            else round(cal_spread, 6),
        }
        _append_log(args.log, log_row)
        print(f"{Fore.GREEN}appended eval log row to: {args.log}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
