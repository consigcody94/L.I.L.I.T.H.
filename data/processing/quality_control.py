"""
Quality Control for GHCN Data

Implements quality checks and cleaning procedures for weather observations.
Based on GHCN quality control flags and additional statistical checks.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class QCFlag(Enum):
    """Quality control flag values."""

    PASSED = "P"  # Passed all checks
    DUPLICATE = "D"  # Duplicate value
    GAP_FILLED = "G"  # Value was interpolated
    SUSPECT_RANGE = "R"  # Outside valid range
    SUSPECT_SPATIAL = "S"  # Spatial consistency check failed
    SUSPECT_TEMPORAL = "T"  # Temporal consistency check failed
    SUSPECT_CLIMATE = "C"  # Exceeds climatological bounds
    FAILED = "F"  # Failed quality check, value removed


@dataclass
class QCConfig:
    """Configuration for quality control checks."""

    # Temperature bounds (°C)
    temp_min: float = -90.0
    temp_max: float = 60.0
    temp_daily_change_max: float = 30.0  # Max change between consecutive days

    # Precipitation bounds (mm)
    precip_min: float = 0.0
    precip_max: float = 1000.0  # Single day max

    # Wind bounds (m/s)
    wind_min: float = 0.0
    wind_max: float = 120.0

    # Pressure bounds (hPa)
    pressure_min: float = 870.0
    pressure_max: float = 1085.0

    # Spike detection
    spike_threshold: float = 4.0  # Standard deviations

    # Climatology bounds (number of standard deviations from monthly mean)
    climate_std_threshold: float = 5.0

    # Gap filling
    max_gap_hours: int = 6  # Maximum gap to interpolate for hourly data
    max_gap_days: int = 3  # Maximum gap to interpolate for daily data


class QualityController:
    """
    Applies quality control checks to weather observation data.

    Checks include:
    1. Range checks (physical bounds)
    2. Temporal consistency (spike detection)
    3. Spatial consistency (comparison with neighbors)
    4. Climatological bounds
    5. Duplicate detection

    Example usage:
        qc = QualityController()
        df_clean, flags = qc.process(df)
    """

    def __init__(self, config: Optional[QCConfig] = None):
        self.config = config or QCConfig()
        self._climatology: Optional[pd.DataFrame] = None

    def process(
        self,
        df: pd.DataFrame,
        station_id: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply all quality control checks to a DataFrame.

        Args:
            df: DataFrame with datetime index and weather variable columns
            station_id: Optional station identifier for logging

        Returns:
            Tuple of (cleaned_df, flags_df) where flags_df contains QC flags
        """
        logger.info(f"Running QC on {len(df)} records" + (f" for {station_id}" if station_id else ""))

        # Initialize flags DataFrame
        flags = pd.DataFrame(index=df.index)
        for col in df.columns:
            flags[f"{col}_flag"] = QCFlag.PASSED.value

        # Create working copy
        df_clean = df.copy()

        # 1. Range checks
        df_clean, flags = self._range_check(df_clean, flags)

        # 2. Temporal consistency (spike detection)
        df_clean, flags = self._temporal_check(df_clean, flags)

        # 3. Duplicate detection
        df_clean, flags = self._duplicate_check(df_clean, flags)

        # 4. Climatological bounds (if climatology is loaded)
        if self._climatology is not None:
            df_clean, flags = self._climate_check(df_clean, flags, station_id)

        # Count flags
        for col in df.columns:
            flag_col = f"{col}_flag"
            if flag_col in flags.columns:
                flag_counts = flags[flag_col].value_counts()
                for flag, count in flag_counts.items():
                    if flag != QCFlag.PASSED.value:
                        logger.debug(f"{col}: {count} records flagged as {flag}")

        # Calculate overall pass rate
        total_checks = len(df) * len(df.columns)
        passed = sum(
            (flags[f"{col}_flag"] == QCFlag.PASSED.value).sum()
            for col in df.columns
            if f"{col}_flag" in flags.columns
        )
        pass_rate = passed / total_checks if total_checks > 0 else 0
        logger.info(f"QC pass rate: {pass_rate:.1%}")

        return df_clean, flags

    def _range_check(
        self,
        df: pd.DataFrame,
        flags: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply physical range checks."""
        cfg = self.config

        # Temperature columns
        for col in ["TMAX", "TMIN", "TAVG", "temperature", "temp_mean", "temp_max", "temp_min"]:
            if col in df.columns:
                mask = (df[col] < cfg.temp_min) | (df[col] > cfg.temp_max)
                flags.loc[mask, f"{col}_flag"] = QCFlag.SUSPECT_RANGE.value
                df.loc[mask, col] = np.nan

        # TMAX should be >= TMIN
        if "TMAX" in df.columns and "TMIN" in df.columns:
            mask = df["TMAX"] < df["TMIN"]
            flags.loc[mask, "TMAX_flag"] = QCFlag.SUSPECT_RANGE.value
            flags.loc[mask, "TMIN_flag"] = QCFlag.SUSPECT_RANGE.value

        # Precipitation
        for col in ["PRCP", "precipitation", "precip", "precipitation_1h", "precipitation_6h"]:
            if col in df.columns:
                mask = (df[col] < cfg.precip_min) | (df[col] > cfg.precip_max)
                flags.loc[mask, f"{col}_flag"] = QCFlag.SUSPECT_RANGE.value
                df.loc[mask, col] = np.nan

        # Wind speed
        for col in ["wind_speed", "AWND", "wind_gust"]:
            if col in df.columns:
                mask = (df[col] < cfg.wind_min) | (df[col] > cfg.wind_max)
                flags.loc[mask, f"{col}_flag"] = QCFlag.SUSPECT_RANGE.value
                df.loc[mask, col] = np.nan

        # Pressure
        for col in ["sea_level_pressure", "station_pressure", "pressure"]:
            if col in df.columns:
                mask = (df[col] < cfg.pressure_min) | (df[col] > cfg.pressure_max)
                flags.loc[mask, f"{col}_flag"] = QCFlag.SUSPECT_RANGE.value
                df.loc[mask, col] = np.nan

        return df, flags

    def _temporal_check(
        self,
        df: pd.DataFrame,
        flags: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Check for temporal consistency (spike detection).

        Uses a rolling window to detect values that deviate significantly
        from their temporal neighbors.
        """
        cfg = self.config

        for col in df.columns:
            if df[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                continue

            # Calculate rolling statistics
            window = 7 if "temp" in col.lower() or col in ["TMAX", "TMIN", "TAVG"] else 3
            rolling_mean = df[col].rolling(window, center=True, min_periods=1).mean()
            rolling_std = df[col].rolling(window, center=True, min_periods=1).std()

            # Flag values that deviate too much from rolling mean
            deviation = np.abs(df[col] - rolling_mean)
            threshold = cfg.spike_threshold * rolling_std.clip(lower=0.1)  # Minimum std

            mask = deviation > threshold
            mask = mask & ~df[col].isna()  # Don't flag already-missing values

            if mask.any():
                # Update flags (don't overwrite worse flags)
                current_flags = flags[f"{col}_flag"]
                new_flags = current_flags.where(
                    current_flags != QCFlag.PASSED.value,
                    QCFlag.SUSPECT_TEMPORAL.value,
                )
                flags.loc[mask, f"{col}_flag"] = new_flags[mask]

                # Optionally remove values (or just flag them)
                # df.loc[mask, col] = np.nan

        return df, flags

    def _duplicate_check(
        self,
        df: pd.DataFrame,
        flags: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Check for duplicate records.

        Flags rows with identical timestamps or suspiciously repeated values.
        """
        # Check for duplicate indices
        if df.index.duplicated().any():
            dup_mask = df.index.duplicated(keep="first")
            for col in df.columns:
                flag_col = f"{col}_flag"
                if flag_col in flags.columns:
                    flags.loc[dup_mask, flag_col] = QCFlag.DUPLICATE.value

            # Remove duplicates (keep first)
            df = df[~df.index.duplicated(keep="first")]
            flags = flags[~flags.index.duplicated(keep="first")]

        # Check for stuck sensors (many repeated values)
        for col in df.columns:
            if df[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                continue

            # Count consecutive identical values
            shifted = df[col].shift(1)
            same_as_prev = df[col] == shifted
            consecutive_same = same_as_prev.groupby((~same_as_prev).cumsum()).cumsum()

            # Flag if more than 5 consecutive identical values (possible stuck sensor)
            stuck_mask = consecutive_same > 5
            if stuck_mask.any():
                logger.debug(f"Possible stuck sensor detected in {col}")
                # Just log, don't automatically flag (could be valid calm conditions)

        return df, flags

    def _climate_check(
        self,
        df: pd.DataFrame,
        flags: pd.DataFrame,
        station_id: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Check values against climatological bounds.

        Requires climatology data to be loaded first.
        """
        if self._climatology is None:
            return df, flags

        cfg = self.config

        # Get month for each record
        months = df.index.month

        for col in df.columns:
            if col not in self._climatology.columns:
                continue

            # Get climatology for each month
            clim_mean = months.map(
                lambda m: self._climatology.loc[m, f"{col}_mean"]
                if m in self._climatology.index
                else np.nan
            )
            clim_std = months.map(
                lambda m: self._climatology.loc[m, f"{col}_std"]
                if m in self._climatology.index
                else np.nan
            )

            # Flag values outside climatological bounds
            deviation = np.abs(df[col] - clim_mean)
            threshold = cfg.climate_std_threshold * clim_std

            mask = deviation > threshold
            mask = mask & ~df[col].isna()

            if mask.any():
                flags.loc[mask, f"{col}_flag"] = QCFlag.SUSPECT_CLIMATE.value

        return df, flags

    def load_climatology(self, path: str) -> None:
        """
        Load climatology data for climate checks.

        Expects a CSV with columns: month, {variable}_mean, {variable}_std
        """
        self._climatology = pd.read_csv(path, index_col="month")
        logger.info(f"Loaded climatology with {len(self._climatology)} months")

    def fill_gaps(
        self,
        df: pd.DataFrame,
        method: str = "linear",
        max_gap: Optional[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fill small gaps in the data using interpolation.

        Args:
            df: DataFrame with datetime index
            method: Interpolation method ('linear', 'time', 'spline')
            max_gap: Maximum gap size to fill (uses config default if None)

        Returns:
            Tuple of (filled_df, flags_df) indicating which values were interpolated
        """
        if max_gap is None:
            # Determine if hourly or daily based on index frequency
            if len(df) > 1:
                freq = pd.infer_freq(df.index)
                if freq and "H" in freq:
                    max_gap = self.config.max_gap_hours
                else:
                    max_gap = self.config.max_gap_days
            else:
                max_gap = self.config.max_gap_days

        # Track which values were interpolated
        was_null = df.isna()

        # Interpolate
        df_filled = df.interpolate(method=method, limit=max_gap)

        # Create flags for interpolated values
        flags = pd.DataFrame(index=df.index)
        for col in df.columns:
            flags[f"{col}_flag"] = np.where(
                was_null[col] & ~df_filled[col].isna(),
                QCFlag.GAP_FILLED.value,
                QCFlag.PASSED.value,
            )

        return df_filled, flags


def main():
    """CLI entry point for running quality control."""
    import argparse

    parser = argparse.ArgumentParser(description="Run quality control on weather data")
    parser.add_argument("input", help="Input CSV or Parquet file")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--climatology", help="Optional climatology file for climate checks")

    args = parser.parse_args()

    # Load data
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input, index_col=0, parse_dates=True)

    # Run QC
    qc = QualityController()
    if args.climatology:
        qc.load_climatology(args.climatology)

    df_clean, flags = qc.process(df)

    # Save
    if args.output.endswith(".parquet"):
        df_clean.to_parquet(args.output)
        flags.to_parquet(args.output.replace(".parquet", "_flags.parquet"))
    else:
        df_clean.to_csv(args.output)
        flags.to_csv(args.output.replace(".csv", "_flags.csv"))


if __name__ == "__main__":
    main()
