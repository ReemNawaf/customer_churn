# src/label_churn.py
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.config import cfg
from src.logger import get_logger

log = get_logger(__name__)

FEATURE_DAYS = cfg["window"]["feature_days"]
TARGET_DAYS = cfg["window"]["target_days"]


def compute_windows(
    df: pd.DataFrame,
    ts_col: str = "ts_dt",
    feature_days: int = FEATURE_DAYS,
    target_days: int = TARGET_DAYS,
):
    """Compute feature window, cut-off day, and target window automatically."""
    min_ts = df[ts_col].min().normalize()  # beginning of first day
    max_ts = df[ts_col].max().normalize()

    # cut-off is the last day of the feature window
    cutoff_date = (min_ts + relativedelta(days=feature_days)) - timedelta(days=1)

    target_start = cutoff_date + timedelta(days=1)
    target_end = target_start + timedelta(days=target_days - 1)

    log.info("Timeline")
    log.info(f"  Earliest event : {min_ts.date()}")
    log.info(f"  Latest event   : {max_ts.date()}")
    log.info(f"  Feature window : {min_ts.date()}  →  {cutoff_date.date()}  (31 days)")
    log.info(f"  Cut-off day    : {cutoff_date.date()}")
    log.info(f"  Target window  : {target_start.date()}  →  {target_end.date()}  (31 days)")

    if target_end > max_ts:
        log.warning("target_end is later than last available data!")

    return cutoff_date, target_start, target_end


def label_churn(
    df: pd.DataFrame,
    user_col: str = "userId",
    ts_col: str = "ts_dt",
    target_start: pd.Timestamp | None = None,
    target_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Label each user as churn(1) or active(0) based on activity in the target window."""

    if target_start is None or target_end is None:
        raise ValueError("Must provide target_start and target_end dates")

    # Aggregate User Last Activity
    user_last = df.groupby(user_col, as_index=False)[ts_col].max().rename(columns={ts_col: "last_ts_dt"})

    # Flag churn if no activity in [target_start, target_end]
    def assign_label(last_ts):
        return 1 if last_ts < target_start else 0

    user_last["churn"] = user_last["last_ts_dt"].apply(assign_label)
    return user_last


if __name__ == "__main__":
    # 1. Load cleaned parquet
    path = f"{Path().resolve()}/{cfg['data']['processed_path']}"
    df = pd.read_parquet(path)

    # 2. Ensure ts_dt is datetime
    df["ts_dt"] = pd.to_datetime(df["ts_dt"], utc=True)

    # 3. Compute windows
    cutoff_date, target_start, target_end = compute_windows(
        df,
        ts_col="ts_dt",
        feature_days=FEATURE_DAYS,
        target_days=TARGET_DAYS,
    )

    # 4. Label churn
    churn_df = label_churn(
        df,
        user_col="userId",
        ts_col="ts_dt",
        target_start=target_start,
        target_end=target_end,
    )

    # 5. Save labeled data
    out_path = f"{Path().resolve()}/{cfg['data']['labeled_path']}"
    churn_df.to_parquet(out_path, index=False)
    log.info(f"Saved churn labels → {out_path}")
    print(churn_df.head())
    print(churn_df["churn"].value_counts())
