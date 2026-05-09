"""
GHCN Daily data processor - converts raw .dly files to training format
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger


class GHCNProcessor:
    """Process GHCN Daily files into training-ready format."""

    # GHCN file format: fixed-width columns
    # ID (11) + Year (4) + Month (2) + Element (4) + 31 * (Value(5) + MFlag(1) + QFlag(1) + SFlag(1))

    ELEMENTS = ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'SNWD']
    MISSING_VALUE = -9999

    def __init__(self, raw_dir: Path, processed_dir: Path, stations_file: Optional[Path] = None):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.stations_file = stations_file
        self.stations_dir = self.raw_dir / "stations"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Load station metadata if available
        self.station_metadata = {}
        if stations_file and stations_file.exists():
            self._load_station_metadata()

    def _load_station_metadata(self):
        """Load station lat/lon from stations file."""
        with open(self.stations_file, 'r') as f:
            for line in f:
                # GHCN stations file format:
                # ID (11) + LAT (9) + LON (10) + ELEV (7) + STATE (3) + NAME (31) + ...
                station_id = line[0:11].strip()
                lat = float(line[12:20].strip())
                lon = float(line[21:30].strip())
                elev = float(line[31:37].strip()) if line[31:37].strip() else 0.0
                name = line[41:71].strip()
                self.station_metadata[station_id] = {
                    'lat': lat,
                    'lon': lon,
                    'elevation': elev,
                    'name': name
                }

    def parse_dly_file(self, filepath: Path) -> pd.DataFrame:
        """Parse a single .dly file into a DataFrame."""
        records = []

        with open(filepath, 'r') as f:
            for line in f:
                if len(line) < 269:  # Minimum valid line length
                    continue

                station_id = line[0:11]
                year = int(line[11:15])
                month = int(line[15:17])
                element = line[17:21]

                if element not in self.ELEMENTS:
                    continue

                # Parse 31 daily values
                for day in range(1, 32):
                    try:
                        start = 21 + (day - 1) * 8
                        value_str = line[start:start+5].strip()
                        mflag = line[start+5:start+6]
                        qflag = line[start+6:start+7]

                        if not value_str:
                            continue

                        value = int(value_str)

                        # Skip missing values and flagged quality issues
                        if value == self.MISSING_VALUE:
                            continue
                        if qflag.strip() not in ['', ' ']:  # Has quality flag
                            continue

                        # Create date
                        try:
                            date = datetime(year, month, day)
                        except ValueError:
                            continue  # Invalid date (e.g., Feb 30)

                        records.append({
                            'station_id': station_id,
                            'date': date,
                            'element': element,
                            'value': value
                        })
                    except (ValueError, IndexError):
                        continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Pivot to get elements as columns
        df = df.pivot_table(
            index=['station_id', 'date'],
            columns='element',
            values='value',
            aggfunc='first'
        ).reset_index()

        # Convert units: temps from tenths of °C, precip from tenths of mm
        if 'TMAX' in df.columns:
            df['TMAX'] = df['TMAX'] / 10.0
        if 'TMIN' in df.columns:
            df['TMIN'] = df['TMIN'] / 10.0
        if 'PRCP' in df.columns:
            df['PRCP'] = df['PRCP'] / 10.0
        if 'SNOW' in df.columns:
            df['SNOW'] = df['SNOW'] / 10.0
        if 'SNWD' in df.columns:
            df['SNWD'] = df['SNWD'] / 10.0

        return df

    def process_all_stations(self, min_years: int = 10) -> pd.DataFrame:
        """Process all station files and combine."""
        all_data = []
        station_files = list(self.stations_dir.glob("*.dly"))

        logger.info(f"Processing {len(station_files)} station files...")

        for i, filepath in enumerate(station_files):
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(station_files)} stations")

            df = self.parse_dly_file(filepath)
            if df.empty:
                continue

            # Check if station has enough data
            years_of_data = (df['date'].max() - df['date'].min()).days / 365
            if years_of_data < min_years:
                continue

            # Add station metadata
            station_id = filepath.stem
            if station_id in self.station_metadata:
                meta = self.station_metadata[station_id]
                df['lat'] = meta['lat']
                df['lon'] = meta['lon']
                df['elevation'] = meta['elevation']

            all_data.append(df)

        if not all_data:
            logger.error("No valid station data found!")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        logger.success(f"Combined {len(combined)} records from {len(all_data)} stations")

        return combined

    def create_training_sequences(
        self,
        df: pd.DataFrame,
        input_days: int = 30,
        target_days: int = 90,
        stride: int = 7,
        features: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create training sequences for the model.

        Args:
            df: DataFrame with processed weather data
            input_days: Number of days of history to use as input
            target_days: Number of days to predict (default 90 to match model horizon)
            stride: Step size between sequences
            features: List of feature columns to use (default ['TMAX','TMIN','PRCP'])

        Returns:
            X: Input sequences [N, input_days, features]
            Y: Target sequences [N, target_days, features]
            meta: Station metadata [N, 4] (lat, lon, elev, day_of_year)
            dates: Forecast start date as np.datetime64[D] [N] — first day of Y window.
                   Use this for chronological train/val splitting to avoid leakage.
        """
        sequences_X = []
        sequences_Y = []
        sequences_meta = []
        sequences_dates = []

        if features is None:
            features = ['TMAX', 'TMIN', 'PRCP']

        # Process each station separately
        stations = df['station_id'].unique()
        logger.info(f"Creating sequences from {len(stations)} stations...")

        for station_id in stations:
            station_df = df[df['station_id'] == station_id].copy()
            station_df = station_df.sort_values('date')

            # Ensure we have required features
            for feat in features:
                if feat not in station_df.columns:
                    station_df[feat] = np.nan

            # Fill missing values with interpolation
            station_df[features] = station_df[features].interpolate(method='linear', limit=7)

            # Drop rows with too many NaN
            station_df = station_df.dropna(subset=features[:2])

            if len(station_df) < input_days + target_days:
                continue

            # Get metadata
            lat = station_df['lat'].iloc[0] if 'lat' in station_df.columns else 0
            lon = station_df['lon'].iloc[0] if 'lon' in station_df.columns else 0
            elev = station_df['elevation'].iloc[0] if 'elevation' in station_df.columns else 0

            # Create sequences
            values = station_df[features].values
            dates = station_df['date'].values

            for i in range(0, len(values) - input_days - target_days, stride):
                X = values[i:i + input_days]
                Y = values[i + input_days:i + input_days + target_days]

                # Skip if too many NaN
                if np.isnan(X).sum() > input_days * len(features) * 0.3:
                    continue
                if np.isnan(Y).sum() > target_days * len(features) * 0.3:
                    continue

                # Fill remaining NaN with mean
                X = np.nan_to_num(X, nan=np.nanmean(X))
                Y = np.nan_to_num(Y, nan=np.nanmean(Y))

                # Forecast start date (first day of target window) — needed for
                # chronological splitting so val/test never sees future training data.
                target_date = pd.Timestamp(dates[i + input_days])
                day_of_year = target_date.dayofyear / 365.0  # Normalize

                sequences_X.append(X)
                sequences_Y.append(Y)
                sequences_meta.append([lat, lon, elev, day_of_year])
                sequences_dates.append(np.datetime64(target_date.date(), 'D'))

        if not sequences_X:
            logger.error("No valid sequences created!")
            empty = np.array([])
            return empty, empty, empty, empty

        X = np.array(sequences_X, dtype=np.float32)
        Y = np.array(sequences_Y, dtype=np.float32)
        meta = np.array(sequences_meta, dtype=np.float32)
        dates = np.array(sequences_dates, dtype='datetime64[D]')

        logger.success(f"Created {len(X)} training sequences")
        logger.info(f"X shape: {X.shape}, Y shape: {Y.shape}, meta shape: {meta.shape}")

        return X, Y, meta, dates

    def save_training_data(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        meta: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ):
        """Save processed training data."""
        output_dir = self.processed_dir / "training"
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "X.npy", X)
        np.save(output_dir / "Y.npy", Y)
        np.save(output_dir / "meta.npy", meta)
        if dates is not None:
            np.save(output_dir / "dates.npy", dates)

        logger.success(f"Saved training data to {output_dir}")

        # Save normalization stats
        stats = {
            'X_mean': X.mean(axis=(0, 1)),
            'X_std': X.std(axis=(0, 1)),
            'Y_mean': Y.mean(axis=(0, 1)),
            'Y_std': Y.std(axis=(0, 1)),
        }
        np.savez(output_dir / "stats.npz", **stats)


def main():
    """Process GHCN data for training."""
    from pathlib import Path

    base_dir = Path(__file__).parent.parent.parent
    raw_dir = base_dir / "data" / "raw" / "ghcn_daily"
    processed_dir = base_dir / "data" / "processed"
    stations_file = raw_dir / "ghcnd-stations.txt"

    processor = GHCNProcessor(raw_dir, processed_dir, stations_file)

    # Process all stations
    df = processor.process_all_stations(min_years=10)

    if df.empty:
        logger.error("No data to process!")
        return

    # Save intermediate CSV for inspection
    df.to_parquet(processed_dir / "ghcn_combined.parquet")
    logger.info(f"Saved combined data to {processed_dir / 'ghcn_combined.parquet'}")

    # Create training sequences (90-day target matches the model's claimed horizon)
    X, Y, meta, dates = processor.create_training_sequences(
        df,
        input_days=30,
        target_days=90,
        stride=7
    )

    if len(X) > 0:
        processor.save_training_data(X, Y, meta, dates)


if __name__ == "__main__":
    main()
