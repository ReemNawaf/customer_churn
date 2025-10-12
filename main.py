import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.logger import get_logger
from src.build_features.features_services import FeatureBuilderService
from src.build_features.preprocess import PreprocessingService
from src.inference_service import ChurnInferenceService

log = get_logger(__name__)

app = FastAPI(title="Churn Prediction API")

preprocessor = PreprocessingService()
feature_builder = FeatureBuilderService()
service = ChurnInferenceService(stage="Production")


class UsersLogs(BaseModel):
    logs: list[dict]


@app.post("/predict")
def predict(users_logs: UsersLogs):
    log.info("prediction service")

    clean_df = preprocessor.parse_raw_events(users_logs.logs)
    log.info(f"\nclean {clean_df}\n")

    features_df = feature_builder.build_all_features(clean_df)
    log.info(f"\nfeatures {features_df}\n")

    preds_df = service.predict(features_df)
    log.info(f"\npreds {preds_df}\n")

    results = [
        {
            "churn_probability": i["churn_proba"],
            "churn_label": int(i["churn_pred"]),
        }
        for _, i in preds_df.iterrows()
    ]

    return {"message": results}
