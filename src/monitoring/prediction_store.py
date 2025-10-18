from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


class PredictionStore:
    """
    Minimal prediction logger.
    - Persists each prediction row with features & model metadata
    - Default backend: SQLite (fast, portable). Optional Parquet mirror.
    """

    def __init__(
        self,
        sqlite_path: str | Path = "data/monitoring/predictions.db",
        table: str = "predictions",
        parquet_dir: Optional[str | Path] = "data/monitoring/parquet",
    ):
        self.sqlite_path = Path(sqlite_path)
        self.table = table
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None

        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        if self.parquet_dir:
            self.parquet_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_schema()

    # ----------------------------
    # Public API
    # ----------------------------
    def log_batch(
        self,
        df_features: pd.DataFrame,
        df_predictions: pd.DataFrame,
        *,
        request_id: str,
        model_name: str,
        model_stage: str,
        model_version: Optional[str] = None,
        model_uri: Optional[str] = None,
        raw_payload: Optional[dict] = None,
        timestamp_utc: Optional[datetime] = None,
    ) -> None:
        """
        Log a batch of predictions.
        - df_features: original feature rows (must include userId)
        - df_predictions: columns ['userId','churn_probability','churn_label']
        """
        print(df_predictions)

        if "userId" not in df_features.columns:
            raise ValueError("df_features must include 'userId' column")
        if not {"userId", "churn_proba", "churn_pred"}.issubset(df_predictions.columns):
            raise ValueError("df_predictions must include userId, churn_probability, churn_label")

        # Merge features + preds on userId (left keeps features order)
        out = df_features.merge(df_predictions, on="userId", how="left").copy()

        # Drop duplicates and suffixes (_x/_y)
        out = out.loc[:, ~out.columns.duplicated()]  # remove duplicate columns
        out.columns = [c.replace("_x", "").replace("_y", "") for c in out.columns]

        # Standardize prediction column names
        out = out.rename(
            columns={
                "churn_proba": "churn_probability",
                "churn_pred": "churn_label",
            }
        )

        ts = (timestamp_utc or datetime.now(timezone.utc)).isoformat()

        # Add metadata columns
        out.insert(0, "timestamp_utc", ts)
        out.insert(1, "request_id", request_id)
        out.insert(2, "model_name", model_name)
        out.insert(3, "model_stage", model_stage)
        out.insert(4, "model_version", model_version or "")
        out.insert(5, "model_uri", model_uri or "")

        # Keep only the required columns for the DB
        # (store everything else inside features_json)
        db_columns = [
            "timestamp_utc",
            "request_id",
            "model_name",
            "model_stage",
            "model_version",
            "model_uri",
            "userId",
            "churn_probability",
            "churn_label",
            "features_json",
            "raw_payload_json",
        ]

        # Build features_json from all columns except the ones above
        meta_cols = set(db_columns) - {"features_json", "raw_payload_json"}
        feature_cols = [c for c in out.columns if c not in meta_cols]
        out["features_json"] = out[feature_cols].apply(lambda r: json.dumps(r.to_dict(), default=str), axis=1)

        # Add raw_payload (same for all rows)
        # out["raw_payload_json"] = json.dumps(raw_payload, default=str) if raw_payload else ""

        # Filter strictly to DB schema columns
        out = out[[c for c in db_columns if c in out.columns]]

        print("out\n", out)

        # Persist clean version to SQLite
        self._append_sqlite(out)

        # Also write a parquet snapshot per request_id (easy offline analysis)
        if self.parquet_dir:
            pq_path = self.parquet_dir / f"{request_id}.parquet"
            out.to_parquet(pq_path, index=False)

    # ----------------------------
    # Internals
    # ----------------------------
    def _ensure_schema(self):
        with sqlite3.connect(self.sqlite_path) as con:
            # Create table if not exists; extra columns will be added dynamically via pandas to_sql
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    timestamp_utc TEXT,
                    request_id TEXT,
                    model_name TEXT,
                    model_stage TEXT,
                    model_version TEXT,
                    model_uri TEXT,
                    userId TEXT,
                    churn_probability REAL,
                    churn_label INTEGER,
                    features_json TEXT,
                    raw_payload_json TEXT
                    -- Note: feature columns will be appended by pandas to_sql if not present
                )
                """
            )
            con.commit()

    def _append_sqlite(self, df: pd.DataFrame) -> None:
        with sqlite3.connect(self.sqlite_path) as con:
            df.to_sql(self.table, con, if_exists="append", index=False)
