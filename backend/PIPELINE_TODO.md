# Pipeline improvement checklist

Simple roadmap from prior discussion. Check items off as you go.

---

## 1. Baselines + bin diagnostics (same split as today)

- **Attribution (in `scripts/eval/eval_baselines.py`):** full logistic prints **coefficients**; **ablations** train on the same split with (a) **price only** `close_ret_3d`, (b) **sentiment only** the three reddit columns. Compare test accuracy: if **full ≈ price_only**, reddit bundle adds little on this slice; if **full drops** without sentiment, reddit matters but stays noisy. Optional columns in `eval_history.csv` track `acc_price_only`, `acc_sentiment_only`, and the two deltas.
- **Milestone A (done):** `had_reddit` in `feature_builder` merge output (`1` if `mention_count >= 1`). In `train_baseline` / eval, **train-constant** columns are skipped in the fit (`active_feature_cols`) so behavior matches pre-A until `had_reddit` varies (e.g. future outer join). Legacy merged CSVs without `had_reddit` default that column to `1` in `prepare_merged_for_training`.
- **Milestone B (done):** merge includes `weighted_sentiment_lag1`, `buzz_lag1`, `weighted_sentiment_roll3_mean`, `weighted_sentiment_roll5_mean`. Baseline **fit** adds **`weighted_sentiment_lag1` only** (roll3+roll5+buzz_lag1 together overfit on current N; they stay in csv for later / ridge). Re-merge then `eval_baselines.py`; compare `eval_history`.
- **Milestone C′ (done):** `buzz_dod = buzz - buzz_lag1` in merge and in `BASELINE_FEATURE_COLS`. Re-merge + `eval_baselines`; compare `eval_history` vs pre‑C′.
- Reuse the same cleaning, sort, `train_lookback_days` window, and train/test split logic as `train_baseline.py` (time-ordered rows, last 20% test).
- Add trivial predictors on **that test set** and record metrics:
  - Always predict up (always long directionally for classification).
  - Always predict down.
  - Majority class on the training window (if that differs from “always up”).
  - Momentum-only rule using existing features (e.g. sign or threshold on `close_ret_3d` alone).
- Compare: accuracy, precision, recall (same definitions as now) vs the logistic model.
- Histogram or deciles of `pred_prob_up_1d` on the test set.
- Count what fraction of test rows have prob below 0.45, between 0.45–0.55, above 0.55.
- Simple calibration table: within each prob bin, mean `y_up_1d` vs mean predicted prob.
- Optional: rank correlation between prob and `y_ret_1d` (does ranking beat random?).

**Done when:** You can say in one glance whether the current model beats trivial rules and how squeezed probabilities are.

**Run:** from `rtsa/backend`, `python scripts/eval/eval_baselines.py` (appends one row to `data/processed/metrics/eval_history.csv`; use `--no-log` to skip).

---

## 2. Feature enrichment + missing-data semantics (Reddit side)

- Treat “no Reddit activity that day” separately from “neutral sentiment” (e.g. flags, NaNs, or imputation instead of silently filling with 0 where it hides missing signal).
- Add rolling sentiment features (e.g. 3-day and 5-day averages) per ticker, using only data available through day **D** for predicting **D+1** (no lookahead).
- Add rolling / change features for buzz or engagement (e.g. change vs prior day, short rolling mean).
- Add mention-side features already in `reddit_daily_all` if not in merge yet (e.g. `mention_count`, `total_engagement` level or transforms), including cross-sectional ideas later (e.g. vs same-day peers).
- Add lagged sentiment (e.g. prior 1–2 days) where it still respects no leakage.
- Wire new fields in `feature_builder.py` and extend `train_baseline.py` `feature_cols` to match.
- Re-run baselines from section 1 after a feature bump (same split contract).

**Done when:** Merged CSV carries richer, clearly documented columns and zeros mean “neutral where we had data,” not “we had no data.”

---

## 3. Walk-forward or rolling retrain (after features settle)

- Freeze a feature schema version (list of columns + definitions) so backtests stay comparable.
- Replace single global split with rolling windows (e.g. train on past N days or N rows, test on next chunk, step forward).
- Log metrics and optional PnL-style summaries **per window** (not only one final split).
- Avoid peeking: each window’s fit uses only data strictly before its test chunk.

**Done when:** You see stability (or drift) across time, not one lucky slice.

---

## 4. Richer model (only after 1–3 justify it)

- Require: logistic regression with enriched features **beats** the section 1 baselines in a stable way across walk-forward windows (not just one run).
- If moving beyond logistic: use strong regularization, watch for overfit on ~1–2k rows, keep validation honest.
- Revisit long/short thresholds only **after** probabilities or ranking actually spread out (otherwise wider thresholds just trade more noise).

**Done when:** A nonlinear model is clearly better than linear + features on the same walk-forward protocol, not only on one static split.

---

## Quick reference


| Phase | Purpose                                                           |
| ----- | ----------------------------------------------------------------- |
| 1     | Prove the model beats dumb rules; measure probability collapse.   |
| 2     | Give the model information worth separating; fix zero vs missing. |
| 3     | Measure robustness as data grows.                                 |
| 4     | Add complexity only when justified.                               |


