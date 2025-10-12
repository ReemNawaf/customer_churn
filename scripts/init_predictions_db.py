import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.monitoring.prediction_store import PredictionStore

if __name__ == "__main__":
    store = PredictionStore("data/monitoring/predictions.db")
    store._ensure_schema()  # force schema creation
    print("Database initialized and table created at:", store.sqlite_path)
