"""
Build user-level feature matrix for churn prediction
from raw, cleaned event logs.

Input:
    - df_logs: pandas DataFrame of cleaned logs
        Expected columns:
        ['ts_dt','userId','sessionId','page','auth','status','level','itemInSession',
         'registration_dt','gender','region','device','artist','song','length', ...]

    - cutoff_date: pandas.Timestamp (the snapshot date at which churn is defined)

Output:
    - features_df: pandas DataFrame
        One row per userId with all engineered features (~30 cols)
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import cfg
from configs.logger import get_logger
from src.build_features.preprocess import PreprocessingService

log = get_logger(__name__)

# -------------------------------------------------------------------
# Top-level pipeline
# -------------------------------------------------------------------


class FeatureBuilderService:
    """Aggregates structured logs into user-level features"""

    def build_all_features(self, df_logs: pd.DataFrame, cutoff_date: pd.Timestamp = None) -> pd.DataFrame:
        """
        Main pipeline: orchestrates all feature groups and returns full user-level matrix.
        """

        if cutoff_date is None:
            cutoff_date = pd.Timestamp.now(tz="UTC")

        # 1. Filter to feature window, to avoid data leakage
        df_feat = df_logs[df_logs["ts_dt"] <= cutoff_date].copy()

        # 2. Build each feature block
        demo = self.build_demographics_features(df_feat, cutoff_date)
        engage = self.build_engagement_features(df_feat, cutoff_date)
        consist = self.build_consistency_features(df_feat, cutoff_date)
        sessions = self.build_session_features(df_feat, cutoff_date)
        diversity = self.build_diversity_features(df_feat, cutoff_date)
        account = self.build_account_features(df_feat, cutoff_date)
        derived = self.build_derived_features(demo, engage, consist, sessions, diversity, account, cutoff_date)

        # 3. Merge all on userId
        features_df = (
            demo.merge(engage, on="userId", how="outer")
            .merge(consist, on="userId", how="outer")
            .merge(sessions, on="userId", how="outer")
            .merge(diversity, on="userId", how="outer")
            .merge(account, on="userId", how="outer")
            .merge(derived, on="userId", how="outer")
        )

        return features_df

    # -------------------------------------------------------------------
    # 1. Cut-Off Date
    # -------------------------------------------------------------------
    def compute_cutoff_date(self, df: pd.DataFrame, date_col: str = "ts_dt") -> pd.Timestamp:
        """
        Automatically pick the cut-off date based on the last full month in the dataset.

        Rules:
        - Find the max timestamp in df[date_col].
        - Determine the first day of that max month.
        - Target window = that month (full month).
        - Cut-off = day before that month starts.

        Example:
        min ts_dt = 2017-11-05
        max ts_dt = 2018-11-24  --> last full month is 2018-10
        cut-off   = 2018-09-30
        target    = 2018-10-01 → 2018-10-31
        """
        max_date = pd.to_datetime(df[date_col].max()).normalize()
        print("max_date", max_date)

        # 1. Last day of that month
        last_day_of_month = max_date + MonthEnd(0)

        # 2. If max_date is NOT the last day → last month is incomplete → step back one month
        if max_date < last_day_of_month:
            # step back to previous month’s first day
            last_month_start = (max_date.replace(day=1) - MonthEnd(1)).replace(day=1)
        else:
            # if month is complete, keep its first day
            last_month_start = max_date.replace(day=1)

        log.info(f"last_month_start {last_month_start}")

        # 3. Cut-off = day before last_month_start
        cutoff_date = last_month_start - pd.Timedelta(days=1)
        log.info(f"cutoff_date {cutoff_date}")

        return cutoff_date

    # -------------------------------------------------------------------
    # 1. Demographics
    # -------------------------------------------------------------------
    def build_demographics_features(self, df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute user-level static demographic features.

        Features:
        1. gender              → most common gender seen for the user
        2. region              → most common region in their logs
        3. device              → most common device in their logs
        4. tenure_days         → (cutoff_date - registration_dt) in days
        5. signup_cohort       → year-month (YYYY-MM) of registration
        6. account_age_group   → categorical bucket of tenure

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned logs containing columns:
                ['userId','gender','region','device','registration_dt', ...]
        cutoff_date : pd.Timestamp
            The snapshot date at which features are calculated.

        Returns
        -------
        pd.DataFrame
            Columns: [userId, gender, region, device,
                    tenure_days, signup_cohort, account_age_group]
        """

        # Defensive copy
        d = df.copy()

        # -------------------------------
        # 1. Reduce to one row per user
        # -------------------------------
        # For categorical features → use the mode (most frequent value per user)
        agg_funcs = {
            "gender": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            "region": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            "device": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            # Keep earliest registration date per user
            "registration_dt": "min",
        }

        demo = d.groupby("userId").agg(agg_funcs).reset_index()

        # -------------------------------
        # 2. Tenure in days
        # -------------------------------
        demo["tenure_days"] = (cutoff_date - demo["registration_dt"]).dt.days.clip(lower=0)  # avoid negatives if cutoff < registration

        # -------------------------------
        # 3. Signup cohort → YYYY-MM
        # -------------------------------
        demo["signup_cohort"] = demo["registration_dt"].dt.tz_localize(None).dt.to_period("M").astype(str)

        # -------------------------------
        # 4. Account age group buckets
        # -------------------------------
        def bucket_age(days: Optional[float]) -> str:
            if pd.isna(days):
                return "unknown"
            if days < 30:
                return "<1m"
            elif days < 90:
                return "1–3m"
            elif days < 180:
                return "3–6m"
            elif days < 365:
                return "6–12m"
            else:
                return "1y+"

        demo["account_age_group"] = demo["tenure_days"].apply(bucket_age)

        # Final column order
        demo = demo[["userId", "gender", "region", "device", "tenure_days", "signup_cohort", "account_age_group"]]

        return demo

    # -------------------------------------------------------------------
    # 2. Engagement
    # -------------------------------------------------------------------
    def build_engagement_features(self, df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute user-level engagement & usage pattern features.

        Features:
        1. recency_days       → days since last activity (cutoff - last active)
        2. freq_7d            → #sessions in last 7 days
        3. freq_30d           → #sessions in last 30 days
        4. freq_90d           → #sessions in last 90 days
        5. active_days_ratio  → (#active days / total days in feature window)
        6. longest_gap_days   → longest gap (in days) between consecutive active days
        7. weekend_ratio      → proportion of sessions on Sat+Sun
        8. night_ratio        → proportion of sessions between 8 PM–6 AM

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned logs containing:
            ['userId','sessionId','ts_dt', ...]
        cutoff_date : pd.Timestamp
            The snapshot date to stop feature window.

        Returns
        -------
        pd.DataFrame
            Columns: [userId, recency_days, freq_7d, freq_30d, freq_90d,
                    active_days_ratio, longest_gap_days, weekend_ratio, night_ratio]
        """

        d = df.copy()
        d = d[d["ts_dt"] <= cutoff_date].copy()

        # Ensure datetime and timezone are set
        d["ts_dt"] = pd.to_datetime(d["ts_dt"], utc=True)

        # -----------------------------------
        # Precompute useful fields
        # -----------------------------------
        d["date"] = d["ts_dt"].dt.date  # daily granularity
        d["dow"] = d["ts_dt"].dt.dayofweek  # 0=Mon,6=Sun
        d["hour"] = d["ts_dt"].dt.hour

        feature_window_days = (cutoff_date.normalize() - d["ts_dt"].min().normalize()).days + 1

        # -----------------------------------
        # A) Recency
        # -----------------------------------
        last_activity = d.groupby("userId")["ts_dt"].max().rename("last_ts_dt")
        recency_days = (cutoff_date - last_activity).dt.days.clip(lower=0)

        # -----------------------------------
        # B) Frequency (#sessions in recent windows)
        # -----------------------------------
        sessions = d.groupby(["userId", "sessionId"])["ts_dt"].min().reset_index()  # each session's start time

        freq_7d = sessions.groupby("userId")["ts_dt"].apply(lambda s: (s >= cutoff_date - pd.Timedelta(days=7)).sum()).rename("freq_7d")  # explicitly select ts_dt

        freq_30d = sessions.groupby("userId")["ts_dt"].apply(lambda g: (g >= cutoff_date - pd.Timedelta(days=30)).sum()).rename("freq_30d")

        freq_90d = sessions.groupby("userId")["ts_dt"].apply(lambda g: (g >= cutoff_date - pd.Timedelta(days=90)).sum()).rename("freq_90d")

        # -----------------------------------
        # C) Active days ratio
        # -----------------------------------
        active_days = d.groupby("userId")["date"].nunique().rename("active_days")
        active_days_ratio = (active_days / feature_window_days).rename("active_days_ratio")

        # -----------------------------------
        # D) Longest inactivity gap
        # -----------------------------------
        def longest_gap(g: pd.Series) -> float:
            days = sorted(pd.to_datetime(g).unique())
            if len(days) <= 1:
                return feature_window_days
            gaps = np.diff(days)  # numpy automatically handles datetime64 diffs
            return gaps.max().days

        longest_gap_days = d.groupby("userId")["date"].apply(longest_gap).rename("longest_gap_days")

        # -----------------------------------
        # E) Weekend ratio
        # -----------------------------------
        weekend_ratio = sessions.groupby("userId")["ts_dt"].apply(lambda s: (s.dt.dayofweek >= 5).sum() / len(s)).rename("weekend_ratio")

        # -----------------------------------
        # F) Night ratio (20:00–06:00)
        # -----------------------------------
        night_ratio = sessions.groupby("userId")["ts_dt"].apply(lambda s: ((s.dt.hour >= 20) | (s.dt.hour < 6)).sum() / len(s)).rename("night_ratio")

        # -----------------------------------
        # Merge all features
        # -----------------------------------
        feats = pd.concat([recency_days, freq_7d, freq_30d, freq_90d, active_days_ratio, longest_gap_days, weekend_ratio, night_ratio], axis=1).reset_index()

        return feats

    # -------------------------------------------------------------------
    # 3. Consistency & Trend
    # -------------------------------------------------------------------
    def build_consistency_features(self, df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute consistency & trend metrics of each user's weekly activity.

        Features:
        1. weekly_mean      → mean weekly sessions in the whole feature window
        2. weekly_std       → std-dev of weekly sessions
        3. weekly_var       → variance of weekly sessions
        4. activity_slope   → trend of weekly activity counts near cutoff
        5. momentum_ratio   → ratio of recent 4-week activity to previous 4-week activity

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned logs containing ['userId','sessionId','ts_dt']
        cutoff_date : pd.Timestamp
            Snapshot date.

        Returns
        -------
        pd.DataFrame
            [userId, weekly_mean, weekly_std, weekly_var,
                    activity_slope, momentum_ratio]
        """

        d = df.copy()
        d = d[d["ts_dt"] <= cutoff_date].copy()

        # Ensure datetime
        d["ts_dt"] = pd.to_datetime(d["ts_dt"], utc=True)

        # -----------------------------------------
        # Prepare weekly session counts
        # -----------------------------------------
        # Collapse to sessions (1 row per session)
        sessions = d.groupby(["userId", "sessionId"])["ts_dt"].min().reset_index()

        # Convert to year-week periods
        sessions["year_week"] = sessions["ts_dt"].dt.tz_convert(None).dt.to_period("W")

        # Count weekly sessions per user
        weekly_counts = sessions.groupby(["userId", "year_week"]).size().reset_index(name="sessions_per_week")

        # -----------------------------------------
        # Compute consistency stats
        # -----------------------------------------
        stats = weekly_counts.groupby("userId")["sessions_per_week"].agg(weekly_mean="mean", weekly_std="std", weekly_var="var").fillna(0)

        # -----------------------------------------
        # Activity slope (trend near cutoff)
        # → fit linear regression on last 4 weeks
        # -----------------------------------------
        def compute_slope(y: pd.Series) -> float:
            if len(y) < 2:
                return 0.0
            x = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(x, y.values)
            return float(model.coef_[0])

        slopes = weekly_counts.groupby("userId")["sessions_per_week"].apply(compute_slope).rename("activity_slope")

        # -----------------------------------------
        # Momentum ratio
        # → recent 2 weeks / previous 2 weeks
        # -----------------------------------------
        def compute_momentum(y: pd.Series) -> float:
            if len(y) == 0:
                return 0.0
            last2 = y.tail(2).sum()
            prev2 = y.tail(4).head(2).sum()
            return round(last2 / prev2, 3) if prev2 > 0 else np.nan

        momentum = weekly_counts.groupby("userId")["sessions_per_week"].apply(compute_momentum).rename("momentum_ratio")

        # -----------------------------------------
        # Combine all features
        # -----------------------------------------
        feats = stats.join(slopes, how="outer").join(momentum, how="outer").reset_index().fillna(0)

        return feats

    # -------------------------------------------------------------------
    # 4. Session-Level
    # -------------------------------------------------------------------
    def build_session_features(self, df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute user-level session-based metrics.

        Features:
        1. total_sessions         → total #sessions for the user
        2. avg_session_len_min    → average session duration in minutes
        3. std_session_len_min    → std-dev of session durations in minutes
        4. total_play_time_min    → total sum of song length played (minutes)
        5. sessions_per_active_day→ avg #sessions per active day

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned logs containing:
            ['userId','sessionId','ts_dt','length',...]
        cutoff_date : pd.Timestamp
            Snapshot date (feature window ends here).

        Returns
        -------
        pd.DataFrame
            Columns:
            [userId, total_sessions,
                    avg_session_len_min,
                    std_session_len_min,
                    total_play_time_min,
                    sessions_per_active_day]
        """

        d = df.copy()
        d = d[d["ts_dt"] <= cutoff_date].copy()
        d["ts_dt"] = pd.to_datetime(d["ts_dt"], utc=True)
        d["date"] = d["ts_dt"].dt.date

        # ---------------------------------------------------------
        # A) Session durations
        # ---------------------------------------------------------
        # Compute session start & end times
        session_stats = d.groupby(["userId", "sessionId"])["ts_dt"].agg(["min", "max"]).rename(columns={"min": "start", "max": "end"})
        session_stats["session_len_min"] = (session_stats["end"] - session_stats["start"]).dt.total_seconds() / 60.0

        # ---------------------------------------------------------
        # B) Aggregate per user
        # ---------------------------------------------------------
        total_sessions = session_stats.groupby("userId").size().rename("total_sessions")

        avg_session_len = session_stats.groupby("userId")["session_len_min"].mean().rename("avg_session_len_min")

        std_session_len = session_stats.groupby("userId")["session_len_min"].std().fillna(0).rename("std_session_len_min")

        # ---------------------------------------------------------
        # C) Total play time (song length)
        # ---------------------------------------------------------
        # 'length' column is in seconds of each song played
        total_play_time = d.groupby("userId")["length"].sum().fillna(0).div(60.0).rename("total_play_time_min")  # convert seconds to minutes

        # ---------------------------------------------------------
        # D) Sessions per active day
        # ---------------------------------------------------------
        active_days = d.groupby("userId")["date"].nunique().rename("active_days")
        sessions_per_active_day = (total_sessions / active_days).replace([np.inf, -np.inf], 0).fillna(0).rename("sessions_per_active_day")

        # ---------------------------------------------------------
        # E) Combine all features
        # ---------------------------------------------------------
        feats = pd.concat(
            [
                total_sessions,
                avg_session_len,
                std_session_len,
                total_play_time,
                sessions_per_active_day,
            ],
            axis=1,
        ).reset_index()

        return feats

    # -------------------------------------------------------------------
    # 5. Diversity
    # -------------------------------------------------------------------
    def build_diversity_features(self, df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute user-level content diversity metrics.

        Features:
        1. unique_pages    → number of unique pages visited
        2. unique_songs    → number of unique songs played
        3. unique_artists  → number of unique artists played
        4. diversity_ratio → unique_songs / total_play_events

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned logs containing columns:
            ['userId','page','song','artist','sessionId','ts_dt', ...]
        cutoff_date : pd.Timestamp
            Snapshot date to limit the feature window.

        Returns
        -------
        pd.DataFrame
            Columns: [userId, unique_pages, unique_songs,
                    unique_artists, diversity_ratio]
        """

        d = df.copy()
        d = d[d["ts_dt"] <= cutoff_date].copy()

        # ------------------------------
        # A) Unique counts
        # ------------------------------
        unique_pages = d.groupby("userId")["page"].nunique().rename("unique_pages")

        unique_songs = d.groupby("userId")["song"].nunique().rename("unique_songs")

        unique_artists = d.groupby("userId")["artist"].nunique().rename("unique_artists")

        # ------------------------------
        # B) Diversity ratio
        # ------------------------------
        # Only count actual play events
        play_events = d.loc[d["page"] == "NextSong"].groupby("userId")["song"].count().rename("play_events")

        diversity_ratio = (unique_songs / play_events).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 1).rename("diversity_ratio")  # ratio bounded between 0 and 1

        # ------------------------------
        # C) Combine features
        # ------------------------------
        feats = pd.concat([unique_pages, unique_songs, unique_artists, diversity_ratio], axis=1).reset_index()

        return feats

    # -------------------------------------------------------------------
    # 6. Account / Level
    # -------------------------------------------------------------------
    def build_account_features(self, df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute user-level account-related metrics.

        Features:
        1. current_level      → most recent subscription level ('free' / 'paid')
        2. level_changes      → number of times user switched level
        3. time_as_paid_days  → total days spent as 'paid'
        4. auth_fail_ratio    → proportion of auth failures (e.g., 'Logout','Error','Cancelled') among all auth events

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned logs containing columns:
            ['userId','level','auth','ts_dt']
        cutoff_date : pd.Timestamp
            Snapshot date to stop feature window.

        Returns
        -------
        pd.DataFrame
            Columns: [userId, current_level, level_changes,
                    time_as_paid_days, auth_fail_ratio]
        """

        d = df.copy()
        d = d[d["ts_dt"] <= cutoff_date].copy()
        d["ts_dt"] = pd.to_datetime(d["ts_dt"], utc=True)
        d = d.sort_values(["userId", "ts_dt"])

        # ------------------------------------------------
        # 1. Current level (last known level before cutoff)
        # ------------------------------------------------
        current_level = d.groupby("userId")["level"].last().rename("current_level")

        # ------------------------------------------------
        # 2. Level changes (count #times level switched)
        # ------------------------------------------------
        def count_level_changes(g: pd.Series) -> int:
            g = g.dropna().astype(str).str.lower()
            if g.empty:
                return 0
            return (g.shift() != g).sum() - 1  # subtract first group

        level_changes = d.groupby("userId")["level"].apply(count_level_changes).rename("level_changes")

        # ------------------------------------------------
        # 3. Time spent as paid (days)
        # ------------------------------------------------
        def calc_time_as_paid(g: pd.DataFrame) -> float:
            g = g.sort_values("ts_dt")
            g["level"] = g["level"].str.lower()
            # Identify all timestamps where user was 'paid'
            paid_periods = g[g["level"] == "paid"]["ts_dt"]
            if paid_periods.empty:
                return 0.0
            return (paid_periods.max() - paid_periods.min()).days + 1

        time_as_paid_days = d.groupby("userId").apply(calc_time_as_paid, include_groups=False).rename("time_as_paid_days")

        # ------------------------------------------------
        # 4. Merge all
        # ------------------------------------------------
        feats = pd.concat([current_level, level_changes, time_as_paid_days], axis=1).reset_index()

        return feats

    # -------------------------------------------------------------------
    # 7. Derived Indicators
    # -------------------------------------------------------------------
    def build_derived_features(
        self, demo: pd.DataFrame, engage: pd.DataFrame, consist: pd.DataFrame, sessions: pd.DataFrame, diversity: pd.DataFrame, account: pd.DataFrame, cutoff_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Compute higher-level features derived by combining previous blocks.

        Features:
        1. engagement_ratio       → freq_30d / active_days_ratio
        2. diversity_per_session  → unique_songs / total_sessions
        3. session_len_change_pct → relative change in session length
                                    (recent 4 weeks vs previous 4 weeks)

        Parameters
        ----------
        demo, engage, consist, sessions, diversity, account : pd.DataFrame
            Individual feature blocks already built.
        cutoff_date : pd.Timestamp
            Snapshot date (not directly used here but kept for consistency).

        Returns
        -------
        pd.DataFrame
            Columns: [userId, engagement_ratio,
                            diversity_per_session,
                            session_len_change_pct]
        """

        # ----------------------------------------------------
        # Merge the needed sources
        # ----------------------------------------------------
        df = demo[["userId"]].copy()

        # Bring in columns needed for derived metrics
        df = (
            df.merge(engage[["userId", "freq_30d", "active_days_ratio"]], on="userId", how="left")
            .merge(sessions[["userId", "total_sessions", "avg_session_len_min"]], on="userId", how="left")
            .merge(diversity[["userId", "unique_songs"]], on="userId", how="left")
        )

        # Fill missing numeric values
        num_cols = ["freq_30d", "active_days_ratio", "total_sessions", "avg_session_len_min", "unique_songs"]
        df[num_cols] = df[num_cols].fillna(0)

        # ----------------------------------------------------
        # 1. Engagement ratio → freq_30d / active_days_ratio
        # ----------------------------------------------------
        df["engagement_ratio"] = (df["freq_30d"] / df["active_days_ratio"].replace(0, np.nan)).fillna(0)

        # ----------------------------------------------------
        # 2. Diversity per session → unique_songs / total_sessions
        # ----------------------------------------------------
        df["diversity_per_session"] = (df["unique_songs"] / df["total_sessions"].replace(0, np.nan)).fillna(0)

        # ----------------------------------------------------
        # 3. Session length change %
        #    (recent 4 weeks avg len vs previous 4 weeks)
        # ----------------------------------------------------
        # To compute properly we need raw session history, but here we approximate:
        # We'll assume the existing avg_session_len_min is the global baseline.
        # This function can be extended if historical weekly data is available.

        # Placeholder: we set change pct to 0 (no change).
        df["session_len_change_pct"] = 0.0

        # ----------------------------------------------------
        # Final output
        # ----------------------------------------------------
        feats = df[["userId", "engagement_ratio", "diversity_per_session", "session_len_change_pct"]].copy()

        return feats


def print_content(path: str):
    # Load the parquet file
    df = pd.read_parquet(path)

    # Show basic info
    print(df.head())  # first 5 rows
    print(df.shape)  # (rows, columns)
    print(df.columns.tolist())  # list of columns


# -------------------------------------------------------------------
# Example CLI execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    obj = f"{Path().resolve()}/{cfg['data']['raw_path']}"
    out_path = obj = f"{Path().resolve()}/{cfg['data']['features_path']}"

    # df_logs = pd.read_parquet(raw_path)
    preprocessor = PreprocessingService()
    feature_builder = FeatureBuilderService()

    df_logs = preprocessor.parse_raw_events(obj)
    cutoff_date = feature_builder.compute_cutoff_date(df_logs)
    features_df = feature_builder.build_all_features(df_logs, cutoff_date)

    features_df.to_parquet(out_path, index=False)
    log.info(f"Features built: {features_df.shape}")

    print_content(out_path)
