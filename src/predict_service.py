import joblib
import pandas as pd

from src.preprocessing.build_features import build_all_features


class ChurnPredictor:
    def __init__(self, model_path="models/churn_model.pkl"):
        self.pipeline = joblib.load(model_path)

    def predict(self, new_logs_df: pd.DataFrame) -> pd.DataFrame:
        # Same feature engineering steps you used in features.py
        # build_all_features(new_logs_df, cutoff_date)
        features = build_all_features(new_logs_df, cutoff_date=pd.Timestamp.now())
        preds = self.pipeline.predict(features.drop(columns=["userId"]))
        probs = self.pipeline.predict_proba(features.drop(columns=["userId"]))[:, 1]
        return pd.DataFrame({"userId": features["userId"], "churn_pred": preds, "churn_prob": probs})


def predict_service(model, X_test):
    y_pred = model.predict(X_test)
    print(y_pred[:4])
    return y_pred
