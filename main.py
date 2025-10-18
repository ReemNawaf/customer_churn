import os
import sys
from uuid import uuid4

from fastapi import FastAPI

from src.models import UsersLogs
from src.monitoring.prediction_store import PredictionStore

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.logger import get_logger
from src.build_features.features_services import FeatureBuilderService
from src.build_features.preprocess import PreprocessingService
from src.inference_service import ChurnInferenceService

log = get_logger(__name__)

app = FastAPI(title="Churn Prediction API")

preprocessor = PreprocessingService()
feature_builder = FeatureBuilderService()
inference_service = ChurnInferenceService(stage="Production")

# Init prediction store
pred_store = PredictionStore(
    sqlite_path="data/monitoring/predictions.db",
    parquet_dir="data/monitoring/parquet",
)


@app.post("/predict_from_logs")
def predict(users_logs: UsersLogs):
    req_id = str(uuid4())
    log.info("prediction service")

    clean_df = preprocessor.parse_raw_events(users_logs.logs)
    log.info(f"\nclean {clean_df}\n")

    features_df = feature_builder.build_features(clean_df)
    log.info(f"\nfeatures {features_df}\n")

    preds_df = inference_service.predict(features_df)
    log.info(f"\npreds {preds_df}\n")

    # Log predictions (+ features + model metadata)
    pred_meta = {
        "request_id": req_id,
        "model_name": inference_service.model_name,
        "model_stage": inference_service.stage,
        "model_version": None,  # optionally fill from registry if you fetch it
        "model_uri": f"models:/{inference_service.model_name}/{inference_service.stage}",
    }
    pred_store.log_batch(
        df_features=features_df.reset_index(drop=True),
        df_predictions=preds_df.reset_index(drop=True),
        request_id=pred_meta["request_id"],
        model_name=pred_meta["model_name"],
        model_stage=pred_meta["model_stage"],
        model_version=pred_meta["model_version"],
        model_uri=pred_meta["model_uri"],
        raw_payload=users_logs.dict(),  # full request for audit
    )

    # Return predictions
    return preds_df.to_dict(orient="records")

    # results = [
    #     {
    #         "churn_probability": i["churn_proba"],
    #         "churn_label": int(i["churn_pred"]),
    #     }
    #     for _, i in preds_df.iterrows()
    # ]

    # return {"message": results}
