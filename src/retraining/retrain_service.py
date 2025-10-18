"""
Retraining Service
------------------
Automates model retraining when drift or performance degradation is detected.

Workflow:
1. Check latest drift report or scheduled trigger
2. If drift exceeds threshold (or scheduled retrain), reload data
3. Run full preprocessing + feature building + training pipeline
4. Compare new model vs. current production model
5. Register the new model in MLflow if it outperforms
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score, roc_auc_score

from src.build_features.features_services import features_df

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from configs.config import cfg
from configs.logger import get_logger
from src.constants import META_COLS
from src.training.model_selection_service import (
    ModelSelectionService,
    mlflow_load_model,
)
from src.training.train_service import ChurnTrainingPipeline

log = get_logger(__name__)
load_dotenv()
os.getenv

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")

# CONFIG
DRIFT_REPORT_DIR = f"{Path().resolve()}/{cfg['report']['folder']}"

DRIFT_THRESHOLD = 3  # number of features with drift before retraining
PERFORMANCE_DROP = 0.05  # acceptable AUC drop before retraining
LOOKBACK_DAYS = 90

DEFAULT_SQLITE = cfg["tables"]["path"]
DEFAULT_TABLE = cfg["tables"]["prediction_table"]


# 1. CHECK DRIFT REPORTS
def check_latest_drift_report(report_dir: str = DRIFT_REPORT_DIR) -> tuple[bool, Path | None]:
    """Return (should_retrain, latest_report_path)"""
    reports = sorted(Path(report_dir).glob("data_drift_*.csv"))
    if not reports:
        log.warning("No drift reports found.")
        return False, None

    latest = reports[-1]
    df = pd.read_csv(latest)
    n_drifted = df.query("status == 'drift'").shape[0]

    log.info(f"Latest drift report: {latest.name}, drifted_features={n_drifted}")
    return n_drifted >= DRIFT_THRESHOLD, latest


# 2. RETRAIN MODEL
def retrain_model(experiment_name, features_df, labels) -> Path:
    log.info("Starting retraining process...")

    trainer = ChurnTrainingPipeline(features_df=features_df, labels_df=labels, experiment_name=experiment_name)
    trainer.train()

    selector = ModelSelectionService(experiment_name=experiment_name)
    runs_df = selector.get_experiment_runs()
    best_run = selector.select_best_model(runs_df)
    reg_result = selector.register_model(best_run, model_name="churn_model")

    log.info(f"Retrained model is registered {reg_result}")

    return reg_result


# 3. EVALUATE & COMPARE
def compare_with_production(
    df_features,
    labels,
    model_name: str = MLFLOW_MODEL_NAME,
) -> bool:
    """
    Compare new model (local file) with the production version in MLflow.
    Returns True if new model outperforms.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    prod_model = mlflow_load_model(model_name=model_name, isProd=True)
    if not prod_model:
        log.warning("No production model found — auto-accept new model.")
        return True
    log.info(f"Comparing against Production model {prod_model}")

    # Load old model from MLflow
    new_model = mlflow_load_model(model_name=model_name, model_version="2")
    if not new_model:
        log.warning("No production model found — auto-accept new model.")
        return True
    log.info(f"Comparing against new model {new_model}")

    # Load evaluation data (recent period)
    X = features_df
    y = labels

    old_preds = prod_model.predict_proba(X)[:, 1]
    new_preds = new_model.predict_proba(X)[:, 1]

    old_auc = roc_auc_score(y, old_preds)
    new_auc = roc_auc_score(y, new_preds)
    old_f1 = f1_score(y, (old_preds > 0.5).astype(int))
    new_f1 = f1_score(y, (new_preds > 0.5).astype(int))

    log.info(f"Old AUC={old_auc:.3f}, New AUC={new_auc:.3f}")
    log.info(f"Old F1={old_f1:.3f}, New F1={new_f1:.3f}")

    return (new_auc - old_auc) > PERFORMANCE_DROP


# 4. REGISTER TO MLFLOW
def register_new_model(model_path: Path, model_name: str = MLFLOW_MODEL_NAME):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Retraining")

    with mlflow.start_run(run_name="retrain_job"):
        mlflow.log_artifact(str(model_path))
        result = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/{model_path.name}",
            name=model_name,
        )
        log.info(f"Model registered: {result.name}, version={result.version}")

        # Promote to Production
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=result.name,
            version=result.version,
            stage="Production",
            archive_existing_versions=True,
        )
        log.info(f"Model promoted to Production: {result.name} v{result.version}")


def load_production_features(
    sqlite_path: str | Path,
    table: str,
) -> pd.DataFrame:
    import sqlite3

    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite not found at {sqlite_path}")

    q = f"""
      SELECT * FROM {table}
    """
    with sqlite3.connect(sqlite_path) as con:
        df = pd.read_sql_query(q, con)

    if df.empty:
        raise RuntimeError("No production rows found in the table.")

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
    features = df.loc[:, cols_to_keep]

    # Coerce dtypes best-effort
    for c in features.columns:
        if c == "userId":
            continue
        if features[c].dtype == "object":
            # try numeric coercion; if many NaNs, keep as object (categorical)
            coerced = pd.to_numeric(features[c], errors="coerce")
            if coerced.notna().mean() > 0.8:
                features[c] = coerced

    # rename churn_label to churn
    df = df.rename(columns={"churn_label": "churn"})

    return features, df[["userId", "churn"]]


# 5. MAIN ORCHESTRATION
def main():
    df_features, labels = load_production_features(DEFAULT_SQLITE, DEFAULT_TABLE)

    experiment_name = "churn-retrain"
    log.info("Checking drift reports...")
    should_retrain, report_path = check_latest_drift_report()

    if not should_retrain:
        log.info("No retraining needed. Drift is within threshold.")
        return

    log.info("Drift threshold exceeded — starting retraining...")
    retrain_model(experiment_name, df_features, labels)

    if compare_with_production(df_features, labels):
        log.info("New model outperforms production — registering...")
        # register_new_model(new_model_path)
    else:
        log.info("New model did not outperform. Keeping current production version.")


if __name__ == "__main__":
    main()
