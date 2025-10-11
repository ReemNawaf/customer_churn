# file: api/app.py
import pandas as pd
from fastapi import FastAPI
from inference_service import ChurnInferenceService
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")
service = ChurnInferenceService(stage="Production")


class UserFeatures(BaseModel):
    features: dict


@app.post("/predict")
def predict(user: UserFeatures):
    df = pd.DataFrame([user.features])
    preds = service.predict(df)
    return {"churn_probability": preds["churn_proba"].iloc[0], "churn_label": int(preds["churn_pred"].iloc[0])}
