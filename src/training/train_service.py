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

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (  # classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import cfg
from configs.logger import get_logger

log = get_logger(__name__)
load_dotenv()
os.getenv

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

MLFLOW_LOGGING = True


# MAIN CLASS
class ChurnTrainingPipeline:
    def __init__(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        target_col: str = "churn",
        test_size: float = 0.2,
        random_state: int = 42,
        experiment_name: str = "Churn",
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessor = None
        self.best_model_name = None

        self.features_df = features_df
        self.labels_df = labels_df

        self.experiment_name = experiment_name
        self.model: Optional[RandomForestClassifier] = None
        self.selector: Optional[SelectFromModel] = None
        self.pipeline: Optional[Pipeline] = None

    # 1. LOAD DATA
    def merge_data(self):
        log.info(f"Features: {self.features_df.shape}, Labels: {self.labels_df.shape}")

        # merge to ensure aligned by userId
        df = self.features_df.merge(self.labels_df, on="userId", how="inner")
        return df

    # 2. SPLIT DATA
    def split_data(self, df: pd.DataFrame):
        if "auth_fail_ratio" in df.columns:
            X = df.drop(columns=["auth_fail_ratio"])

        X = df.drop(columns=["userId", self.target_col])
        y = df[self.target_col]

        # Handle boolean columns before detecting dtypes
        bool_cols = X.select_dtypes(include="bool").columns
        if len(bool_cols) > 0:
            log.info(f"Converting boolean columns: {list(bool_cols)} → int")
            X[bool_cols] = X[bool_cols].astype(int)

        # detect categoricals
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

        for col in cat_cols:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

        return X_train, X_test, y_train, y_test, cat_cols, num_cols

    # 3. BUILD PREPROCESSOR
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

    # 4. Selector + Candidate Models
    def build_selector(self) -> SelectFromModel:
        rf_selector = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=self.random_state, n_jobs=-1)
        self.selector = SelectFromModel(estimator=rf_selector, threshold="median")
        return self.selector

    def get_candidate_models(self):
        param = {
            "RandomForest": {
                "n_estimators": 300,
                "class_weight": "balanced",
                "random_state": self.random_state,
                "n_jobs": -1,
            },
            "GradientBoosting": {
                "random_state": self.random_state,
            },
            "LogisticRegression": {
                "max_iter": 500,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "random_state": self.random_state,
            },
            "XGBoost": {
                "n_estimators": 400,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "logloss",
                "random_state": self.random_state,
                "enable_categorical": True,
                "use_label_encoder": False,
            },
            "LightGBM": {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbose": -1,
                "class_weight": "balanced",
                "random_state": self.random_state,
            },
        }

        def get_pipe(model):
            return Pipeline(
                [
                    ("preprocessor", self.preprocessor),
                    ("selector", self.selector),
                    ("clf", model),
                ]
            )

        return {
            "RandomForest": {
                "pipe": get_pipe(RandomForestClassifier(**param["RandomForest"])),
                "param": param["RandomForest"],
            },
            "GradientBoosting": {
                "pipe": get_pipe(GradientBoostingClassifier(**param["GradientBoosting"])),
                "param": param["GradientBoosting"],
            },
            "LogisticRegression": {
                "pipe": get_pipe(LogisticRegression(**param["LogisticRegression"])),
                "param": param["LogisticRegression"],
            },
            "XGBoost": {
                "pipe": XGBClassifier(**param["XGBoost"]),
                "param": param["XGBoost"],
            },
            "LightGBM": {
                "pipe": get_pipe(LGBMClassifier(**param["LightGBM"])),
                "param": param["LightGBM"],
            },
        }

    # 5. Compare Models
    def evaluate_models(self, X_train, y_train, X_test, y_test) -> pd.DataFrame:
        results = []
        candidates = self.get_candidate_models()

        for name, item in candidates.items():
            pipe = item["pipe"]
            param = item["param"]

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

            acc = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)

            metrics = {
                "test_auc": auc_score,
                "test_f1": f1,
                "test_accuracy": acc,
            }

            # Plot ROC curve
            roc_cur_path = f"reports/{name}_roc_curve.png"
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {self.best_model_name}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(roc_cur_path)

            # Plot Confusion Matrix
            con_mat_path = f"reports/{name}_confusion_matrix.png"
            plt.figure(figsize=(4, 4))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {self.best_model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(con_mat_path)
            plt.close()

            if MLFLOW_LOGGING:
                self.mlfow_log(name, param, metrics, pipe, roc_cur_path, con_mat_path, X_train, y_train)

            metrics = {
                "pipe": name,
                **metrics,
            }
            results.append(metrics)

        results_df = pd.DataFrame(results).sort_values(by="test_accuracy", ascending=False)
        log.info(f"\nModel Comparison:\n {results_df}")
        return results_df

    # TRAIN
    def train(self):
        log.info("Load prepared features")
        df = self.merge_data()
        X_train, X_test, y_train, y_test, cat_cols, num_cols = self.split_data(df)

        log.info("Building preprocessing pipeline...")
        self.build_preprocessor(cat_cols, num_cols)
        self.build_selector()

        # Compare candidate models
        results_df = self.evaluate_models(X_train, y_train, X_test, y_test)
        print(results_df)

        # Pick the best model
        best_model_name = results_df.iloc[0]["pipe"]
        log.info(f"The best model is: {best_model_name}")

        # rebuild the best pipeline for full training
        best_candidate = self.get_candidate_models()[best_model_name]["pipe"]
        best_candidate.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

        self.pipeline = best_candidate
        self.best_model_name = best_model_name

        # self.print_selected_features()

        log.info("Training complete.")
        # self.evaluate(X_test, y_test, X_train.columns)

    # ML-Flow Functions
    def mlfow_log(self, name, params, metrics, pipe, roc_cur_path, con_mat_path, X_train, y_train):
        input_example = X_train.sample(5)
        log.info(f"Logging {name}: Parameters, Metrics, and Model")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=f"{name}_training_run"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            if "XGB" in name:
                (mlflow.xgboost.log_model(xgb_model=pipe, name=name, input_example=input_example, signature=mlflow.models.infer_signature(X_train, y_train)),)
            else:
                (mlflow.sklearn.log_model(sk_model=pipe, name=name, input_example=input_example, signature=mlflow.models.infer_signature(X_train, y_train)),)

            # log artificates as pictures
            mlflow.log_artifact(roc_cur_path)
            mlflow.log_artifact(con_mat_path)
            mlflow.log_artifact(con_mat_path)
            X_train.to_parquet("mlartifacts/training_stats.parquet")
            mlflow.log_artifact("mlartifacts/training_stats.parquet")

            # TODO: model artificates is stored however not shown to the ui

    # Utility to Print Selected Features
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

    def save_model(self, out_path: str = "mlartifacts/models/churn_model.pkl") -> str:
        """
        Persist the trained pipeline (preprocessor + selector + classifier)
        as a .pkl file for local loading or deployment (non-MLflow use).
        """

        if self.pipeline is None:
            raise RuntimeError("No pipeline trained yet. Call `train()` first.")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, out_path)
        log.info(f"[Model Saved] Trained pipeline stored at: {out_path.resolve()}")

        return str(out_path.resolve())


#   RUN EXAMPLE
if __name__ == "__main__":
    log.info("Loading data...")
    features_path = f"{Path().resolve()}/{cfg['data']['features_path']}"
    labels_path = f"{Path().resolve()}/{cfg['data']['labeled_path']}"
    feature_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)

    trainer = ChurnTrainingPipeline(
        features_path=feature_df,
        labels_path=labels_df,
    )

    trainer.train()
    # trainer.save_model("models/churn_pipeline.pkl")
