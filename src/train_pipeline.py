"""
Churn Prediction Training Pipeline
----------------------------------
Steps:
1. Load pre-built feature matrix + labels
2. Split train / validation / test
3. Preprocess:
     - handle missing
     - encode categoricals
     - scale numeric
4. Feature selection
5. Train classifier
6. Evaluate + save artifacts
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import cfg
from src.logger import get_logger

log = get_logger(__name__)


# ==========================================================
#                       MAIN CLASS
# ==========================================================
class ChurnTrainingPipeline:
    def __init__(
        self,
        features_path: str,
        labels_path: str,
        target_col: str = "churn",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessor = None
        self.best_model_name = None

        self.feature_df: Optional[pd.DataFrame] = None
        self.labels_df: Optional[pd.DataFrame] = None
        self.model: Optional[RandomForestClassifier] = None
        self.selector: Optional[SelectFromModel] = None
        self.pipeline: Optional[Pipeline] = None

    # ------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------
    def load_data(self):
        log.info("Loading data...")
        self.feature_df = pd.read_parquet(self.features_path)
        self.labels_df = pd.read_parquet(self.labels_path)
        log.info(f"Features: {self.feature_df.shape}, Labels: {self.labels_df.shape}")

        # merge to ensure aligned by userId
        df = self.feature_df.merge(self.labels_df, on="userId", how="inner")
        return df

    # ------------------------------------------------------
    # 2. SPLIT DATA
    # ------------------------------------------------------
    def split_data(self, df: pd.DataFrame):
        X = df.drop(columns=["userId", self.target_col])
        y = df[self.target_col]

        # detect categoricals
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

        return X_train, X_test, y_train, y_test, cat_cols, num_cols

    # ------------------------------------------------------
    # 3. BUILD PREPROCESSOR
    # ------------------------------------------------------
    def build_preprocessor(self, cat_cols: List[str], num_cols: List[str]):
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_pipe, cat_cols),
                ("num", num_pipe, num_cols),
            ]
        )
        return self.preprocessor

    # ---------------------------------------------------
    # 4. Selector + Candidate Models
    # ---------------------------------------------------
    def build_selector(self) -> SelectFromModel:
        rf_selector = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=self.random_state, n_jobs=-1)
        self.selector = SelectFromModel(estimator=rf_selector, threshold="median")
        return self.selector

    def get_candidate_models(self):
        return {
            "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=self.random_state, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(random_state=self.random_state),
            "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs", random_state=self.random_state),
            "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=self.random_state),
            "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbose=-1, class_weight="balanced", random_state=self.random_state),
        }

    # ---------------------------------------------------
    # 5. Compare Models
    # ---------------------------------------------------
    def compare_models(self, X_train, y_train, X_val, y_val) -> pd.DataFrame:
        results = []
        candidates = self.get_candidate_models()

        for name, model in candidates.items():
            pipe = Pipeline(
                [
                    ("preprocessor", self.preprocessor),
                    ("selector", self.selector),
                    ("clf", model),
                ]
            )

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            y_proba = pipe.predict_proba(X_val)[:, 1] if hasattr(pipe, "predict_proba") else None

            metrics = {
                "Model": name,
                "AUC": roc_auc_score(y_val, y_proba) if y_proba is not None else None,
                "F1": f1_score(y_val, y_pred),
                "Accuracy": accuracy_score(y_val, y_pred),
            }
            results.append(metrics)

        results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
        log.info(f"\nModel Comparison:\n {results_df}")
        return results_df

    # ---------------------------------------------------
    # 6. Full Train on Best Model
    # ---------------------------------------------------
    def train_best_model(self, X_train_full, y_train_full, best_model_name: str):
        candidates = self.get_candidate_models()
        best_model = candidates[best_model_name]

        self.pipeline = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                ("selector", self.selector),
                ("clf", best_model),
            ]
        )
        self.pipeline.fit(X_train_full, y_train_full)
        self.best_model_name = best_model_name
        log.info(f"Best model retrained on full training set: {best_model_name}")

    # ------------------------------------------------------
    # 7. EVALUATE
    # ------------------------------------------------------
    def evaluate(self, X_test, y_test, orig_cols: List[str]):
        log.info("Evaluating model...")
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        log.info("\nClassification Report:")
        log.info(classification_report(y_test, y_pred))

        log.info(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
        log.info(f"ROC-AUC: {roc_auc_score(y_test, y_prob)}")
        log.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")

        # Print selected features
        selector = self.pipeline.named_steps["selector"]
        support_mask = selector.get_support()

        log.info("\nSelected Features (after preprocessing):")
        log.info(f"Total selected: {support_mask.sum()} of {len(support_mask)}")

    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------
    def train(self):
        log.info("Load prepared features")
        df = self.load_data()
        X_train, X_test, y_train, y_test, cat_cols, num_cols = self.split_data(df)

        log.info("Building preprocessing pipeline...")
        self.build_preprocessor(cat_cols, num_cols)
        self.build_selector()

        # Compare candidate models
        results_df = self.compare_models(X_train, y_train, X_test, y_test)
        print(results_df)

        # Pick the best model
        best_model_name = results_df.iloc[0]["Model"]

        # Retrain on full training data
        log.info(f"Training the best model ({best_model_name})...")
        self.train_best_model(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), best_model_name)

        self.print_selected_features()

        log.info("Training complete.")
        self.evaluate(X_test, y_test, X_train.columns)

    # ---------------------------------------------------
    # Utility to Print Selected Features
    # ---------------------------------------------------
    def print_selected_features(self):
        # Get feature names after preprocessing
        all_features = self.pipeline.named_steps["preprocessor"].get_feature_names_out()

        # Get boolean mask of selected features
        selected_mask = self.pipeline.named_steps["selector"].get_support()

        # Keep only selected ones
        selected_features = all_features[selected_mask]

        log.info("Selected features:")
        log.info(f"{len(selected_features)} features were selected out of {len(all_features)}")
        log.info(f"{selected_features}")

    # ------------------------------------------------------
    # SAVE PIPELINE
    # ------------------------------------------------------
    def save_model(self, out_path: str = "models/churn_model.pkl"):
        import joblib

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, out_path)
        log.info(f"Pipeline saved to {out_path}")


# ==========================================================
#                     RUN EXAMPLE
# ==========================================================
if __name__ == "__main__":

    features_path = f"{Path().resolve()}/{cfg['data']['features_path']}"
    labels_path = f"{Path().resolve()}/{cfg['data']['labeled_path']}"

    trainer = ChurnTrainingPipeline(
        features_path=features_path,
        labels_path=labels_path,
    )

    trainer.train()
    # trainer.save_model("models/churn_pipeline.pkl")
