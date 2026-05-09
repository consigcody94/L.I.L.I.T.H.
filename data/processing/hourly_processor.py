"""
Processor for sub-daily observations -> training sequences.

Bridges between the ASOS 1-min downloader (which lands hourly-aggregated
parquet) and the same X/Y/meta/dates training format the daily processor
emits. The model architecture is unchanged; we're just feeding it richer
features per token.

Two modes:
    * 'daily':    one token per day, but with 12 channels (4 quartile-of-day
                  temps, day max/min, gust max, mean wind, MSLP min, precip
                  total, precip max-rate, RH min). This is the "easy upgrade"
                  path — same SimpleLILITH, no architecture changes.
    * 'hourly':   one token per hour. Sequences are 720-token (30-day) inputs
                  predicting 2160-token (90-day) outputs. Much heavier — only
                  attempt with the full LILITH-Tiny on the 5070.

For commercial weather products (severe thunderstorm risk, frost timing, gust
forecasts) the daily-with-rich-channels mode tends to be the sweet spot — it
exposes diurnal-cycle and gust statistics that TMAX/TMIN aggregation discards.
Schulz & Lerch (arXiv:2106.09512) found that adding meteorological predictor
variables beyond the target variable significantly improved gust postprocessing
skill at hourly cadence; sub-daily features should help here for similar
reasons. The exact RMSE lift depends on your data and target variables —
*measure it on a held-out year before quoting numbers to users*.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# Map of feature name -> (source column in hourly parquet, aggregation).
# Source columns come from asos_1min.aggregate_to_hourly() -> '<var>_<stat>'.
DAILY_FEATURE_MAP = [
    ("t_00z",       "tmpc_mean",    "hour_eq_0"),
    ("t_06z",       "tmpc_mean",    "hour_eq_6"),
    ("t_12z",       "tmpc_mean",    "hour_eq_12"),
    ("t_18z",       "tmpc_mean",    "hour_eq_18"),
    ("tmax",        "tmpc_max",     "max"),
    ("tmin",        "tmpc_min",     "min"),
    ("rh_min",      "relh_min",     "min"),
    ("wind_mean",   "sknt_mean",    "mean"),
    ("gust_max",    "gust_max",     "max"),
    ("mslp_min",    "mslp_min",     "min"),
    ("prcp_total",  "p01i_mean",    "sum"),  # 1-hr precip sums to 24-hr total
    ("prcp_max",    "p01i_max",     "max"),
]


class HourlyProcessor:
    """Convert hourly parquet into the same (X, Y, meta, dates) tuple the
    daily processor produces, but with richer per-day channels."""

    def __init__(self, hourly_parquet: Path | str):
        self.path = Path(hourly_parquet)
        if not self.path.exists():
            raise FileNotFoundError(f"Hourly parquet not found: {self.path}")
        self.df = pd.read_parquet(self.path)
        if "timestamp" not in self.df.columns:
            raise ValueError("Expected 'timestamp' column from asos_1min.aggregate_to_hourly()")
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        self.df = self.df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    def to_daily_features(self, station_meta: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Roll up hourly -> daily with the multi-channel feature set above.

        station_meta (optional): columns [station, latitude, longitude, elevation].
        Without it, lat/lon/elev fall back to NaN and per-station defaults are 0.
        """
        df = self.df.copy()
        df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.date.astype("datetime64[ns]")
        df["hour"] = df["timestamp"].dt.hour

        rows: List[dict] = []
        for (station, date), group in df.groupby(["station", "date"]):
            row = {"station_id": station, "date": date}
            hourly_subset = group.set_index("hour")

            for out_name, src_col, agg in DAILY_FEATURE_MAP:
                if src_col not in group.columns:
                    row[out_name] = np.nan
                    continue
                if agg.startswith("hour_eq_"):
                    target_h = int(agg.split("_")[-1])
                    row[out_name] = (
                        float(hourly_subset.loc[target_h, src_col])
                        if target_h in hourly_subset.index
                        else np.nan
                    )
                elif agg == "mean":
                    row[out_name] = float(group[src_col].mean(skipna=True))
                elif agg == "max":
                    row[out_name] = float(group[src_col].max(skipna=True))
                elif agg == "min":
                    row[out_name] = float(group[src_col].min(skipna=True))
                elif agg == "sum":
                    row[out_name] = float(group[src_col].sum(skipna=True))
            rows.append(row)

        daily = pd.DataFrame(rows).sort_values(["station_id", "date"]).reset_index(drop=True)

        if station_meta is not None and not station_meta.empty:
            sm = station_meta.rename(columns={"stid": "station_id"})
            daily = daily.merge(sm[["station_id", "latitude", "longitude", "elevation"]],
                                on="station_id", how="left")
        else:
            for col in ("latitude", "longitude", "elevation"):
                daily[col] = np.nan

        return daily

    def create_training_sequences(
        self,
        daily: pd.DataFrame,
        input_days: int = 30,
        target_days: int = 90,
        stride: int = 7,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Same contract as GHCNProcessor.create_training_sequences."""
        feature_cols = [name for name, *_ in DAILY_FEATURE_MAP]
        X_list, Y_list, meta_list, date_list = [], [], [], []

        # Forward-fill within station up to a week, then drop the rest.
        daily = daily.copy()
        daily[feature_cols] = daily.groupby("station_id")[feature_cols].ffill(limit=7)

        for station, sdf in daily.groupby("station_id"):
            sdf = sdf.sort_values("date").reset_index(drop=True)
            if len(sdf) < input_days + target_days:
                continue

            values = sdf[feature_cols].values.astype(np.float32)
            dates = sdf["date"].values
            lat = float(sdf["latitude"].iloc[0]) if not pd.isna(sdf["latitude"].iloc[0]) else 0.0
            lon = float(sdf["longitude"].iloc[0]) if not pd.isna(sdf["longitude"].iloc[0]) else 0.0
            elev = float(sdf["elevation"].iloc[0]) if not pd.isna(sdf["elevation"].iloc[0]) else 0.0

            for i in range(0, len(values) - input_days - target_days, stride):
                X = values[i:i + input_days]
                Y = values[i + input_days:i + input_days + target_days]

                # Reject windows where missingness would silently nudge the model
                # toward the impute rather than the true signal.
                X_nan_frac = np.isnan(X).sum() / X.size
                Y_nan_frac = np.isnan(Y).sum() / Y.size
                if X_nan_frac > 0.3 or Y_nan_frac > 0.3:
                    continue

                X = np.nan_to_num(X, nan=np.nanmean(X) if not np.all(np.isnan(X)) else 0.0)
                Y = np.nan_to_num(Y, nan=np.nanmean(Y) if not np.all(np.isnan(Y)) else 0.0)

                target_date = pd.Timestamp(dates[i + input_days])
                day_of_year = target_date.dayofyear / 365.0

                X_list.append(X)
                Y_list.append(Y)
                meta_list.append([lat, lon, elev, day_of_year])
                date_list.append(np.datetime64(target_date.date(), "D"))

        if not X_list:
            empty = np.array([])
            return empty, empty, empty, empty

        return (
            np.stack(X_list).astype(np.float32),
            np.stack(Y_list).astype(np.float32),
            np.array(meta_list, dtype=np.float32),
            np.array(date_list, dtype="datetime64[D]"),
        )

    @property
    def feature_names(self) -> List[str]:
        return [name for name, *_ in DAILY_FEATURE_MAP]
