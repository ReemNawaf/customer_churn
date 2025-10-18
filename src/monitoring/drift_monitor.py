from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from dotenv import load_dotenv

from configs.config import cfg
from configs.logger import get_logger
from src.constants import META_COLS

log = get_logger(__name__)
load_dotenv()
os.getenv

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

MLFLOW_EXP = "monitoring_experiment"
DEFAULT_SQLITE = cfg["tables"]["path"]
DEFAULT_TABLE = cfg["tables"]["prediction_table"]
DEFAULT_REPORT_DIR = cfg["report"]["folder"]
DEFAULT_BASELINE_FEATURES = cfg["data"]["features_path"]
DEFAULT_BASELINE_SNAPSHOT = "mlartifacts/training_stats.parquet"  # optional saved baseline
NUM_KS_THRESHOLD = 0.2
CAT_PSI_THRESHOLD = 0.25
MIN_SAMPLE = 2


@dataclass
class DriftThresholds:
    ks_stat: float = NUM_KS_THRESHOLD
    psi: float = CAT_PSI_THRESHOLD


def load_recent_production_features(
    sqlite_path: str | Path,
    table: str,
    lookback_days: int,
    min_rows: int = MIN_SAMPLE,
) -> pd.DataFrame:
    import sqlite3

    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite not found at {sqlite_path}")

    since_iso = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    q = "SELECT * FROM " + table + " WHERE timestamp_utc >= ?"
    with sqlite3.connect(sqlite_path) as con:
        df = pd.read_sql_query(q, con, params=[since_iso])

    if df.empty:
        raise RuntimeError("No recent production rows found in the lookback window.")

    # Infer feature columns (everything that is not metadata or label/preds)
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # if the expanded features aren't in the table, expand features_json instead
    if not feature_cols and "features_json" in df.columns:
        try:
            feats_expanded = pd.json_normalize(df["features_json"].dropna().apply(json.loads))
            df = pd.concat([df.reset_index(drop=True), feats_expanded.reset_index(drop=True)], axis=1)
            feature_cols = feats_expanded.columns.tolist()
            print(f"[DriftMonitor] Expanded {len(feature_cols)} features from features_json")
        except Exception as e:
            raise RuntimeError(f"Failed to expand features_json: {e}")

    # if still nothing, raise a clear error
    if not feature_cols:
        raise RuntimeError("No feature columns found or reconstructed from features_json")

    # Return only features (and userId if you want to keep it for debugging)
    cols_to_keep = ["userId"] + feature_cols if "userId" in df.columns else feature_cols
    out = df.loc[:, cols_to_keep]

    # Coerce dtypes best-effort
    for c in out.columns:
        if c == "userId":
            continue
        if out[c].dtype == "object":
            # try numeric coercion; if many NaNs, keep as object (categorical)
            coerced = pd.to_numeric(out[c], errors="coerce")
            if coerced.notna().mean() > 0.8:
                out[c] = coerced
    if len(out) < min_rows:
        raise RuntimeError(f"Too few production rows ({len(out)}). Need at least {min_rows}.")
    print("out", out)
    return out


def load_baseline_features(
    baseline_snapshot_path: Optional[str | Path],
    baseline_features_path: Optional[str | Path],
    min_rows: int = MIN_SAMPLE,
) -> pd.DataFrame:
    """
    Load a training baseline:
      - Prefer a saved snapshot (parquet) of *training features* (recommended),
      - Else fall back to the original training features parquet.
    """
    # Prefer explicit snapshot
    if baseline_snapshot_path and Path(baseline_snapshot_path).exists():
        df = pd.read_parquet(baseline_snapshot_path)
        # If it's stats, not raw rows, we can't do distribution tests. Expect rows.
        # So users should save a sample or the full training features.
        if "userId" in df.columns:
            return df
        else:
            raise RuntimeError(f"{baseline_snapshot_path} exists but doesn't look like raw feature rows. Provide a baseline features parquet.")

    # Fallback to the full training features parquet
    if baseline_features_path and Path(baseline_features_path).exists():
        df = pd.read_parquet(baseline_features_path)
        return df

    raise FileNotFoundError("No valid baseline found. Provide --baseline-snapshot (rows) or --baseline-features (training features parquet).")


def split_numeric_categorical(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Treat anything non-numeric as categorical/string
    cat_cols = [c for c in df.columns if c not in num_cols and c != "userId"]
    return num_cols, cat_cols


def ks_drift(train_s: pd.Series, prod_s: pd.Series) -> dict:
    # Drop NaNs
    a = pd.to_numeric(train_s, errors="coerce").dropna()
    b = pd.to_numeric(prod_s, errors="coerce").dropna()
    if len(a) < MIN_SAMPLE or len(b) < MIN_SAMPLE:
        return {"ks_stat": np.nan, "p_value": np.nan}
    stat, p = ks_2samp(a, b, alternative="two-sided", mode="auto")
    return {"ks_stat": float(stat), "p_value": float(p)}


def psi_categorical(train_s: pd.Series, prod_s: pd.Series, eps: float = 1e-6) -> float:
    # Normalize to string categories for robustness
    t = train_s.astype("string").fillna("__NA__")
    p = prod_s.astype("string").fillna("__NA__")

    # Frequency distributions
    t_counts = t.value_counts(dropna=False)
    p_counts = p.value_counts(dropna=False)

    # Shared category universe
    cats = sorted(set(t_counts.index).union(set(p_counts.index)))
    t_dist = np.array([t_counts.get(c, 0) for c in cats], dtype=float)
    p_dist = np.array([p_counts.get(c, 0) for c in cats], dtype=float)

    # Convert to proportions
    t_prop = t_dist / max(t_dist.sum(), eps)
    p_prop = p_dist / max(p_dist.sum(), eps)

    # PSI
    psi = np.sum((t_prop - p_prop) * np.log((t_prop + eps) / (p_prop + eps)))
    return float(psi)


def build_drift_report(
    train_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    thresholds: DriftThresholds,
) -> pd.DataFrame:
    # Align columns
    common_cols = [c for c in train_df.columns if c in prod_df.columns and c != "userId"]
    if not common_cols:
        raise RuntimeError("No overlapping feature columns between baseline and production.")

    train = train_df[common_cols].copy()
    prod = prod_df[common_cols].copy()

    num_cols, cat_cols = split_numeric_categorical(train)

    rows = []

    # Numeric: KS
    for c in num_cols:
        ks = ks_drift(train[c], prod[c])
        drift = (not math.isnan(ks["ks_stat"])) and (ks["ks_stat"] >= thresholds.ks_stat) and (ks["p_value"] is not None and ks["p_value"] < 0.05)
        rows.append(
            {
                "feature": c,
                "type": "numeric",
                "metric": "KS",
                "value": ks["ks_stat"],
                "p_value": ks["p_value"],
                "status": "drift" if drift else "ok",
                "notes": "",
            }
        )

    # Categorical: PSI
    for c in cat_cols:
        val = psi_categorical(train[c], prod[c])
        drift = (not math.isnan(val)) and (val >= thresholds.psi)
        rows.append(
            {
                "feature": c,
                "type": "categorical",
                "metric": "PSI",
                "value": val,
                "p_value": np.nan,
                "status": "drift" if drift else "ok",
                "notes": "",
            }
        )

    report = pd.DataFrame(rows).sort_values(["status", "metric", "feature"])
    return report


def save_report(report: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"data_drift_{ts}.csv"
    report.to_csv(out_path, index=False)
    return out_path


def log_to_mlflow(report: pd.DataFrame, tracking_uri: Optional[str], experiment: Optional[str]) -> None:
    if mlflow is None:
        return
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)

    drifted = report.query("status == 'drift'").shape[0]
    with mlflow.start_run(run_name="data_drift_monitor"):
        mlflow.log_metric("num_drifted_features", float(drifted))
        # aggregate by type
        for t, df in report.groupby("type"):
            mlflow.log_metric(f"num_drifted_{t}", float(df.query("status == 'drift'").shape[0]))
        # attach the CSV
        tmp = "reports/data_drift_latest.csv"
        report.to_csv(tmp, index=False)
        mlflow.log_artifact(tmp)


def main():
    lookback_days = 7

    # Load data
    prod = load_recent_production_features(DEFAULT_SQLITE, DEFAULT_TABLE, lookback_days)
    log.info(f"Loaded production window: {prod.shape}")

    train = load_baseline_features(DEFAULT_BASELINE_SNAPSHOT, DEFAULT_BASELINE_FEATURES)
    log.info(f"Loaded baseline features: {train.shape}")

    # Build report
    thresholds = DriftThresholds(ks_stat=NUM_KS_THRESHOLD, psi=CAT_PSI_THRESHOLD)
    report = build_drift_report(train, prod, thresholds)
    out_path = save_report(report, DEFAULT_REPORT_DIR)
    log.info(f"Drift report saved to: {out_path}")

    log_to_mlflow(report, MLFLOW_TRACKING_URI, MLFLOW_EXP)
    log.info("Drift summary logged to MLflow")


if __name__ == "__main__":
    main()
