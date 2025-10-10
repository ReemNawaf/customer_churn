"""
Model Selection & Registration Service
--------------------------------------
- Queries MLflow experiments
- Finds best performing model based on metrics
- Registers it in the MLflow Model Registry
- Optionally promotes to 'Staging' or 'Production'
"""

import os
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from configs.config import cfg
from configs.logger import get_logger

log = get_logger(__name__)

MLFLOW_LOGGING = True


class ModelSelectionService:
    def __init__(self, experiment_name: str = "Churn", tracking_uri: str = "http://127.0.0.1:5000"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

    # ---------------------------------------------------
    #  Get all runs from the experiment
    # ---------------------------------------------------
    def get_experiment_runs(self) -> pd.DataFrame:
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found")

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["metrics.test_auc DESC"],  # sort by best AUC
        )

        runs_df = pd.DataFrame(
            [
                {
                    "run_id": r.info.run_id,
                    "model_name": r.data.tags.get("mlflow.runName"),
                    "test_auc": r.data.metrics.get("test_auc"),
                    "test_f1": r.data.metrics.get("test_f1"),
                    "test_accuracy": r.data.metrics.get("test_accuracy"),
                    "artifact_uri": r.info.artifact_uri,
                }
                for r in runs
            ]
        )

        log.info(f"Found {len(runs_df)} runs in experiment '{self.experiment_name}'")
        log.info(runs_df.head())
        return runs_df

    # ---------------------------------------------------
    #  Load certain model
    # ---------------------------------------------------
    def mlflow_load_model(model_name, model_version):
        model_uri = f"models:/{model_name}/{model_version}"

        if "XGB" in model_name:
            loaded_model = mlflow.xgboost.load_model(model_uri)
        else:
            loaded_model = mlflow.sklearn.load_model(model_uri)

        return loaded_model

    # ---------------------------------------------------
    #  Pick the best model (highest AUC)
    # ---------------------------------------------------
    def select_best_model(self, runs_df: pd.DataFrame):
        best_run = runs_df.sort_values("test_auc", ascending=False).iloc[0]
        log.info(f"Best model: {best_run['model_name']} (AUC={best_run['test_auc']:.4f})")
        return best_run

    # ---------------------------------------------------
    # Register the model in MLflow Registry
    # ---------------------------------------------------
    def register_model(self, best_run, model_name="churn_model"):
        model_uri = f"runs:/{best_run['run_id']}/{best_run['model_name']}"
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        log.info(f"Registered model '{model_name}' as version {result.version}")

        # model_uri = f"runs:/{run_id}/model_name"

        # with mlflow.start_run(run_id=run_id):
        #     mlflow.register_model(model_uri=model_uri, name=model_name)

        return result

    # ---------------------------------------------------
    # Optionally promote to Staging/Production
    # ---------------------------------------------------
    def promote_model(self, model_name: str, version: int, stage: str = "Production"):
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )

        # move_model_to_production
        # current_model_uri = f"models:/{model_name}@challenger"

        # client = mlflow.MlflowClient()
        # client.copy_model_version(src_model_uri=current_model_uri, dst_name=production_model_name)

        log.info(f"Model '{model_name}' promoted to stage: {stage}")


# ==========================================================
# RUN EXAMPLE
# ==========================================================
if __name__ == "__main__":
    selector = ModelSelectionService(experiment_name="Churn")

    runs_df = selector.get_experiment_runs()
    best_run = selector.select_best_model(runs_df)
    reg_result = selector.register_model(best_run, model_name="churn_model")
    selector.promote_model("churn_model", reg_result.version, stage="Staging")
