"""
Inference Service for Churn Prediction
--------------------------------------
Loads the latest Production model from MLflow Model Registry,
and serves predictions for new (already processed & feature-ready) data.
"""

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import cfg
from configs.logger import get_logger
from training.model_selection_service import mlflow_load_model

log = get_logger(__name__)
load_dotenv()
os.getenv
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


class ChurnInferenceService:
    def __init__(self, model_name: str = "churn_model", tracking_uri: str = MLFLOW_TRACKING_URI, stage: str = "production"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = model_name
        self.stage = stage
        self.model = mlflow_load_model(self.model_name)

    # ---------------------------------------------------
    # Predict churn for incoming data
    # ---------------------------------------------------
    def predict(self, df: pd.DataFrame):
        """
        Input:  df (processed + feature-ready DataFrame)
        Output: df with churn_proba, churn_pred
        """
        print("df", df)

        preds_proba = self.model.predict_proba(df)[:, 1]
        preds_label = (preds_proba > 0.5).astype(int)

        result = df.copy()
        result["churn_proba"] = preds_proba
        result["churn_pred"] = preds_label
        return result


# ==========================================================
# RUN EXAMPLE (batch prediction)
# ==========================================================
if __name__ == "__main__":
    # TODO: add the pipeline for input (pre, feat)

    # Load processed + feature-ready data (simulate)
    features_path = f"{Path().resolve()}/{cfg['data']['features_path']}"
    predictions_path = f"{Path().resolve()}/{cfg['data']['predictions_path']}"

    data = pd.read_parquet(features_path)

    # Create inference service instance
    service = ChurnInferenceService()

    # Run prediction
    preds = service.predict(data)
    print(preds.head())

    # Save output
    preds.to_parquet(predictions_path, index=False)
    print("Predictions saved to data/predictions/")
