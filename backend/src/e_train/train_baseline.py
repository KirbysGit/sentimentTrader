from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from colorama import Fore, Style
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.utils.path_config import processed_metrics_dir
from src.utils.config import train_lookback_days


@dataclass
class TrainResult:
    ok: bool
    report_path: Path
    model_path: Path
    metrics: Dict[str, float]


def _time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    # simple time-based split (last X% of rows as test).
    n = len(df)
    cut = max(1, int(n * (1.0 - test_frac)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def train_baseline(merged_features_path: Path, test_frac: float = 0.2) -> TrainResult:

    # create the processed metrics directory if it doesn't exist.
    processed_metrics_dir.mkdir(parents=True, exist_ok=True)

    # get the stem of the merged features path.
    stem = Path(merged_features_path).stem.replace("merged_features_", "")

    # create the report path.
    report_path = processed_metrics_dir / f"training_report_{stem}.csv"

    # load the merged features csv file.
    df = pd.read_csv(merged_features_path)

    # basic cleaning.
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["ticker", "date"])

    df["y_ret_1d"] = pd.to_numeric(df["y_ret_1d"], errors="coerce")
    df = df.dropna(subset=["y_ret_1d"]).copy()

    # create the binary direction label.
    df["y_up_1d"] = (df["y_ret_1d"] > 0).astype(int)

    # create the feature columns.
    feature_cols = ["weighted_sentiment", "buzz", "sentiment_chg_1d", "close_ret_3d"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # sort the dataframe by date and ticker.
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # optional rolling window (train on last N days to keep runs bounded).
    if train_lookback_days and train_lookback_days > 0:
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=int(train_lookback_days))
        df = df[df["date"] >= cutoff].copy()

    # split the dataframe into train and test sets.
    train_df, test_df = _time_split(df, test_frac=test_frac)

    # train the model.
    model = LogisticRegression(max_iter=2000)
    model.fit(train_df[feature_cols], train_df["y_up_1d"])
    prob_up = model.predict_proba(test_df[feature_cols])[:, 1]
    pred = (prob_up >= 0.5).astype(int)

    # --- signal thresholds (more conservative than 0.5)
    long_threshold = 0.60
    short_threshold = 0.40

    # calculate the metrics.
    y_true = test_df["y_up_1d"].astype(int).values
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, pred))
    metrics["precision"] = float(precision_score(y_true, pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, pred, zero_division=0))

    # training metadata (so artifacts are self-describing).
    train_start = train_df["date"].min()
    train_end = train_df["date"].max()
    test_start = test_df["date"].min()
    test_end = test_df["date"].max()
    train_up = int(train_df["y_up_1d"].sum())
    train_down = int((train_df["y_up_1d"] == 0).sum())
    test_up = int(test_df["y_up_1d"].sum())
    test_down = int((test_df["y_up_1d"] == 0).sum())

    # write the report (test rows only).
    report = test_df[["ticker", "date", "y_ret_1d", "y_up_1d"] + feature_cols].copy()
    report["pred_up_1d"] = pred
    report["pred_prob_up_1d"] = prob_up

    # trading signal (three-way)
    report["signal"] = "hold"
    report.loc[report["pred_prob_up_1d"] >= long_threshold, "signal"] = "long"
    report.loc[report["pred_prob_up_1d"] <= short_threshold, "signal"] = "short"

    # confidence bucket.
    report["confidence_bucket"] = "low"
    report.loc[(report["pred_prob_up_1d"] >= long_threshold) | (report["pred_prob_up_1d"] <= short_threshold), "confidence_bucket"] = "high"
    report.loc[
        ((report["pred_prob_up_1d"] >= 0.55) & (report["pred_prob_up_1d"] < long_threshold))
        | ((report["pred_prob_up_1d"] > short_threshold) & (report["pred_prob_up_1d"] <= 0.45)),
        "confidence_bucket",
    ] = "medium"

    # pnl proxy : long => y_ret_1d, short => -y_ret_1d, hold => 0
    report["pnl_proxy"] = 0.0
    report.loc[report["signal"] == "long", "pnl_proxy"] = report.loc[report["signal"] == "long", "y_ret_1d"]
    report.loc[report["signal"] == "short", "pnl_proxy"] = -report.loc[report["signal"] == "short", "y_ret_1d"]

    report.to_csv(report_path, index=False)

    # save model artifact (so big runs are not wasted).
    models_dir = processed_metrics_dir.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"baseline_logreg_{stem}_{run_id}.joblib"
    latest_path = models_dir / "baseline_logreg_latest.joblib"

    artifact = {
        "model_type": "logistic_regression",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_cols": feature_cols,
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "train_lookback_days": int(train_lookback_days),
        "rows": {
            "total_rows_after_cleaning": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "date_range": {
            "train_start": train_start.isoformat() if pd.notna(train_start) else None,
            "train_end": train_end.isoformat() if pd.notna(train_end) else None,
            "test_start": test_start.isoformat() if pd.notna(test_start) else None,
            "test_end": test_end.isoformat() if pd.notna(test_end) else None,
        },
        "class_balance": {
            "train_up": train_up,
            "train_down": train_down,
            "test_up": test_up,
            "test_down": test_down,
        },
        "metrics": metrics,
        "model": model,
    }
    joblib.dump(artifact, model_path)
    joblib.dump(artifact, latest_path)

    # print the report path and metrics.
    print(f"{Fore.CYAN}stage 5 - wrote training report to: {Style.RESET_ALL}{report_path.name}")
    print(f"{Fore.CYAN}stage 5 - saved model to: {Style.RESET_ALL}{model_path.name}")
    print(f"{Fore.CYAN}stage 5 - metrics: {Style.RESET_ALL}{metrics}")

    return TrainResult(ok=True, report_path=report_path, model_path=model_path, metrics=metrics)


