import os
import sys
from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import cfg


def profile(df):
    pr = ProfileReport(df, title="Churn Data Profile")
    pr.to_file(cfg["data"]["profile_report"])
    print("----")


if __name__ == "__main__":
    path = f"{Path().resolve()}/{cfg['data']['processed_path']}"
    df = pd.read_parquet(path)
    profile(df)
