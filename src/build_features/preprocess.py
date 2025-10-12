from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import cfg
from configs.logger import get_logger

log = get_logger(__name__)

Jsonish = str | Path | list[dict] | Iterable[dict] | pd.DataFrame

OS_PATTERNS = [
    (r"Windows NT", "Windows"),
    (r"Mac OS X|Macintosh", "macOS"),
    (r"iPhone|iPad|iOS", "iOS"),
    (r"Android", "Android"),
    (r"Linux", "Linux"),
]

BROWSER_PATTERNS = [
    (r"Chrome/", "Chrome"),
    (r"CriOS/", "Chrome (iOS)"),
    (
        r"Safari/",
        "Safari",
    ),  # note: appears alongside Chrome; see special-case below
    (r"Firefox/", "Firefox"),
    (r"Edg/", "Edge"),
]


class PreprocessingService:
    def parse_raw_events(self, obj: Jsonish) -> pd.DataFrame:
        """
        Read raw event logs and return a clean,
        typed DataFrame with normalized columns.

        Supported inputs:
        - Path or str to a JSON Lines file (one JSON per line)
        - Iterable[List[dict]] of events
        - pandas.DataFrame

        Returns a DataFrame with (at least) these columns:
        - ts (int), ts_dt (datetime[UTC])
        - registration (int|None), registration_dt (datetime[UTC]|NaT)
        - userId (str), sessionId (int), page (str), auth (str|None),
        status (Int64|None)
        - level (category), gender (category|None)
        - artist, song, length (float|None)
        - userAgent (str|None), os (str|None), browser (str|None),
        device (str|Desktop/Mobile/Tablet|None)
        - location (str|None), city (str|None), region (str|None)
            country (str|None)
        """

        log.info("Loading raw data")
        df = self.to_dataframe(obj)
        log.info(f"Loaded shape: {df.head()}")
        log.info(f"Loaded shape: {df.shape}")

        # --- Canonicalize column names (some datasets vary in case) ---
        df = df.rename(columns={c: c.strip() for c in df.columns})

        log.info("Preprocessing the data...")
        # --- Type casting of core fields ---
        # Timestamps (ms → UTC datetime)
        if "ts" in df.columns:
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
            df["ts_dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        else:
            df["ts_dt"] = pd.NaT

        if "registration" in df.columns:
            df["registration"] = pd.to_numeric(df["registration"], errors="coerce").astype("Int64")
            df["registration_dt"] = pd.to_datetime(df["registration"], unit="ms", utc=True)
        else:
            df["registration_dt"] = pd.NaT

        # IDs
        if "userId" in df.columns:
            df["userId"] = df["userId"].astype(str).str.strip()
            # Drop empty userId rows if any (some datasets have blanks)
            df = df[df["userId"].str.len() > 0]
        else:
            df["userId"] = pd.Series(dtype="string")

        if "sessionId" in df.columns:
            df["sessionId"] = pd.to_numeric(df["sessionId"], errors="coerce").astype("Int64")

        # Simple string cols

        for col in (
            "page",
            "auth",
            "level",
            "gender",
            "artist",
            "song",
            "location",
            "userAgent",
        ):
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip()

        # HTTP status
        if "status" in df.columns:
            df["status"] = pd.to_numeric(df["status"], errors="coerce").astype("Int64")

        # Length (song length in seconds)
        if "length" in df.columns:
            df["length"] = pd.to_numeric(df["length"], errors="coerce")

        # Normalize `level`/`gender` to categories when present
        if "level" in df.columns:
            df["level"] = df["level"].str.lower().replace({"paid": "paid", "free": "free"}).astype("category")
        if "gender" in df.columns:
            df["gender"] = df["gender"].str.upper().replace({"M": "M", "F": "F"}).astype("category")

        # --- Parse user agent into os / browser / device ---
        ua_series = df["userAgent"] if "userAgent" in df.columns else pd.Series([], dtype="string")
        parsed = ua_series.fillna("").apply(self.parse_user_agent)
        if not parsed.empty:
            df[["os", "browser", "device"]] = pd.DataFrame(parsed.tolist(), index=df.index)

        # --- Normalize location into city / region / country ---
        loc_series = df["location"] if "location" in df.columns else pd.Series([], dtype="string")
        normalized_loc = loc_series.fillna("").apply(self.normalize_location)
        if not normalized_loc.empty:
            df[["city", "region", "country"]] = pd.DataFrame(normalized_loc.tolist(), index=df.index)

        # Sort by time (optional but handy)
        if "ts_dt" in df.columns:
            df = df.sort_values("ts_dt").reset_index(drop=True)

        log.info("Preprocessed the data")

        return df

    def save_processed(self, df: pd.DataFrame):
        proc_path = Path(cfg["data"]["processed_path"])
        proc_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(proc_path, index=False)
        log.info(f"Saved processed data → {proc_path}")

    # ---------- Helpers ----------
    def to_dataframe(self, obj: Jsonish) -> pd.DataFrame:
        """Normalize various inputs to a DataFrame."""

        if isinstance(obj, pd.DataFrame):
            return obj.copy()

        if isinstance(obj, str | Path):
            path = Path(obj)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
            # Assume JSON Lines (one record per line)
            return pd.read_json(path, dtype=False)

        # Iterable of dicts
        if isinstance(obj, list | tuple) or hasattr(obj, "__iter__"):
            # Convert generators/iterables of dicts to list before DataFrame
            return pd.DataFrame(list(obj))

        raise TypeError("Unsupported input type for parse_raw_events().")

    def parse_user_agent(self, ua: str) -> tuple[str | None, str | None, str | None]:
        """
        Very light-weight UA parser (no external deps):
        - OS: Windows/macOS/iOS/Android/Linux
        - Browser: Chrome/Safari/Firefox/Edge
        - Device: Desktop / Mobile / Tablet (heuristics)
        """
        if not ua:
            return None, None, None

        os_name = self.match_first(ua, OS_PATTERNS)
        browser = self.match_first(ua, BROWSER_PATTERNS)

        # If both Chrome and Safari matched, prefer Chrome (common on macOS/iOS)
        if "Chrome" in (browser or "") and "Safari" in ua and browser != "Safari":
            pass  # keep Chrome
        elif browser == "Safari" and "Chrome/" in ua:
            browser = "Chrome"

        # Device heuristic
        ua_l = ua.lower()
        if any(k in ua_l for k in ["iphone", "android", "mobile"]):
            device = "Mobile"
        elif "ipad" in ua_l or "tablet" in ua_l:
            device = "Tablet"
        else:
            device = "Desktop"

        return os_name, browser, device

    def match_first(self, text: str, patterns: list[tuple[str, str]]) -> str | None:
        for pat, label in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                return label
        return None

    def normalize_location(self, loc: str) -> tuple[str | None, str | None, str | None]:
        """
        Normalize a 'City, ST' or 'City, State' or 'City, Country' string into (city, region, country).
        Heuristics:
        - If the second token is two uppercase letters (e.g., TX, CA), assume US state → country='USA'
        - Else: treat as 'City, Country' (region=None)
        """
        if not loc:
            return None, None, None

        parts = [p.strip() for p in loc.split(",") if p.strip()]
        if len(parts) == 1:
            return parts[0], None, None
        if len(parts) >= 2:
            city, rest = parts[0], parts[1]
            # Two-letter uppercase region → assume USA
            if re.fullmatch(r"[A-Z]{2}", rest):
                return city, rest, "USA"
            # If rest looks like a US state name (rough heuristic), keep it as region, country=USA
            if rest.lower() in {
                "alabama",
                "alaska",
                "arizona",
                "arkansas",
                "california",
                "colorado",
                "connecticut",
                "delaware",
                "florida",
                "georgia",
                "hawaii",
                "idaho",
                "illinois",
                "indiana",
                "iowa",
                "kansas",
                "kentucky",
                "louisiana",
                "maine",
                "maryland",
                "massachusetts",
                "michigan",
                "minnesota",
                "mississippi",
                "missouri",
                "montana",
                "nebraska",
                "nevada",
                "new hampshire",
                "new jersey",
                "new mexico",
                "new york",
                "north carolina",
                "north dakota",
                "ohio",
                "oklahoma",
                "oregon",
                "pennsylvania",
                "rhode island",
                "south carolina",
                "south dakota",
                "tennessee",
                "texas",
                "utah",
                "vermont",
                "virginia",
                "washington",
                "west virginia",
                "wisconsin",
                "wyoming",
            }:
                return city, rest, "USA"
            # Otherwise assume it's a country name
            return city, None, rest

        return None, None, None


if __name__ == "__main__":
    obj = f"{Path().resolve()}/{cfg['data']['raw_path']}"
    preprocessor = PreprocessingService()

    df_clean = preprocessor.parse_raw_events(obj)
    preprocessor.save_processed(df_clean)
