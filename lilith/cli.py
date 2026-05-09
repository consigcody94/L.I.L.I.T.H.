"""Unified ``lilith`` command-line interface.

Replaces the scattered ``python -m data.download.X`` invocations with a
discoverable namespaced CLI. Every subcommand is a thin shim around the
existing modules — no logic moves, just routing.

Examples:
    lilith download ghcn --max-stations 500
    lilith download climate
    lilith process ghcn
    lilith train simple --epochs 50 --target-days 90
    lilith train full --variant tiny --target-days 90
    lilith forecast --lat 40.7128 --lon -74.006 --days 14 --ensemble 30
    lilith api --port 8000

The pyproject [project.scripts] entry point ``lilith = "lilith.cli:app"``
is what registers this as a console script after ``pip install -e .``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import typer

# Make the project root importable regardless of where the CLI is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


app = typer.Typer(
    add_completion=False,
    help="L.I.L.I.T.H. — Long-range Intelligent Learning for Integrated Trend Hindcasting.",
    no_args_is_help=True,
)
download_app = typer.Typer(no_args_is_help=True, help="Pull weather data from public sources.")
process_app = typer.Typer(no_args_is_help=True, help="Process raw data into training format.")
train_app = typer.Typer(no_args_is_help=True, help="Train models.")
app.add_typer(download_app, name="download")
app.add_typer(process_app, name="process")
app.add_typer(train_app, name="train")


# ----- download --------------------------------------------------------------


@download_app.command("ghcn")
def download_ghcn(
    output_dir: str = typer.Option("data/raw/ghcn_daily"),
    country: Optional[str] = typer.Option(None, help="2-letter code, e.g. US"),
    min_years: int = typer.Option(30, help="Min years of record per station"),
    max_stations: Optional[int] = typer.Option(None, help="Cap station count"),
    stations_only: bool = typer.Option(False, help="Skip observation downloads"),
):
    """Download GHCN-Daily station archives from NOAA NCEI."""
    from data.download.ghcn_daily import GHCNDailyDownloader

    with GHCNDailyDownloader(output_dir=output_dir) as dl:
        dl.download_stations()
        dl.download_inventory()
        if not stations_only:
            dl.download_all(country=country, min_years=min_years, max_stations=max_stations)


@download_app.command("ghcn-hourly")
def download_ghcn_hourly(
    output_dir: str = typer.Option("data/raw/ghcn_hourly"),
    country: Optional[str] = typer.Option(None),
    min_years: int = typer.Option(20),
    max_stations: Optional[int] = typer.Option(None),
    start_year: int = typer.Option(2000),
    end_year: int = typer.Option(2023),
):
    """Download GHCN-Hourly (ISD-Lite) data."""
    from data.download.ghcn_hourly import GHCNHourlyDownloader

    with GHCNHourlyDownloader(output_dir=output_dir) as dl:
        dl.download_station_list()
        dl.download_all(
            country=country,
            min_years=min_years,
            max_stations=max_stations,
            years=list(range(start_year, end_year + 1)),
        )


@download_app.command("climate")
def download_climate(
    indices: str = typer.Option("enso,nao,pdo,mjo,ao",
                                help="Comma-separated subset (or 'all')"),
    output_dir: str = typer.Option("data/raw/climate"),
    force: bool = typer.Option(False),
    start: str = typer.Option("1950-01-01", help="Start of daily forward-filled frame"),
):
    """Download teleconnection indices (ENSO, NAO, PDO, MJO, AO, …)."""
    from data.download.climate_indices import ClimateIndexDownloader, SOURCES

    keys = list(SOURCES.keys()) if indices == "all" else [k.strip() for k in indices.split(",")]
    dl = ClimateIndexDownloader(output_dir)
    dfs = dl.download(keys, force=force)
    if dfs:
        dl.save_daily_parquet(dfs)
    else:
        typer.echo("No indices downloaded.", err=True)
        raise typer.Exit(1)


@download_app.command("asos-1min")
def download_asos_1min(
    stations: str = typer.Argument(..., help="Comma-separated ICAO IDs (e.g. KORD,KJFK)"),
    start: str = typer.Option(..., help="YYYY-MM"),
    end: str = typer.Option(..., help="YYYY-MM"),
    output_dir: str = typer.Option("data/raw/asos_1min"),
    aggregate: bool = typer.Option(True, help="Run hourly aggregation step after download"),
):
    """Download 1-minute ASOS observations from NCEI v1."""
    from data.download.asos_1min import ASOS1MinDownloader

    sids = [s.strip().upper() for s in stations.split(",") if s.strip()]
    dl = ASOS1MinDownloader(output_dir)
    paths = dl.download(sids, start, end)
    if aggregate and paths:
        dl.aggregate_to_hourly(paths)


@download_app.command("synoptic")
def download_synoptic(
    stations: str = typer.Argument(..., help="Comma-separated Synoptic STIDs"),
    start: str = typer.Option(..., help="YYYYMMDDHHMM UTC"),
    end: str = typer.Option(..., help="YYYYMMDDHHMM UTC"),
    output_dir: str = typer.Option("data/raw/synoptic"),
    token: Optional[str] = typer.Option(None,
                                        help="Synoptic token (or set $SYNOPTIC_TOKEN)"),
):
    """Download Synoptic Mesonet/RAWS/CWOP observations."""
    from data.download.synoptic import SynopticConfig, SynopticDownloader, get_token

    tok = get_token(token)
    if not tok:
        typer.echo(
            "No Synoptic token. Pass --token, set $SYNOPTIC_TOKEN, or write it to "
            "~/.synoptic_token (sign up at https://customer.synopticdata.com/).",
            err=True,
        )
        raise typer.Exit(2)

    sids = [s.strip() for s in stations.split(",") if s.strip()]
    dl = SynopticDownloader(SynopticConfig(token=tok), output_dir=output_dir)
    dl.timeseries(sids, start, end)


# ----- process ---------------------------------------------------------------


@process_app.command("ghcn")
def process_ghcn(
    raw_dir: str = typer.Option("data/raw/ghcn_daily"),
    processed_dir: str = typer.Option("data/processed"),
    min_years: int = typer.Option(10),
    input_days: int = typer.Option(30),
    target_days: int = typer.Option(90, help="90-day horizon matches the model claim"),
    stride: int = typer.Option(7),
):
    """Process raw GHCN-Daily files into X/Y/meta/dates training arrays."""
    from data.processing.ghcn_processor import GHCNProcessor

    raw = Path(raw_dir)
    proc = GHCNProcessor(raw, Path(processed_dir), raw / "ghcnd-stations.txt")
    df = proc.process_all_stations(min_years=min_years)
    if df.empty:
        typer.echo("No GHCN data found.", err=True)
        raise typer.Exit(1)
    df.to_parquet(Path(processed_dir) / "ghcn_combined.parquet")
    X, Y, meta, dates = proc.create_training_sequences(
        df, input_days=input_days, target_days=target_days, stride=stride
    )
    if len(X) == 0:
        typer.echo("No training sequences created.", err=True)
        raise typer.Exit(1)
    proc.save_training_data(X, Y, meta, dates)


@process_app.command("hourly")
def process_hourly(
    parquet: str = typer.Argument(..., help="Path to hourly parquet from asos_1min aggregator"),
    output_dir: str = typer.Option("data/processed/training_hourly"),
    input_days: int = typer.Option(30),
    target_days: int = typer.Option(90),
    stride: int = typer.Option(7),
):
    """Convert hourly parquet to enriched-daily training sequences."""
    import numpy as np

    from data.processing.hourly_processor import HourlyProcessor

    proc = HourlyProcessor(parquet)
    daily = proc.to_daily_features()
    # Caller is responsible for joining station_meta with lat/lon/elev — without
    # that join the trainer will see all-NaN coords, so warn loudly.
    daily["latitude"] = daily.get("latitude", float("nan"))
    daily["longitude"] = daily.get("longitude", float("nan"))
    daily["elevation"] = daily.get("elevation", float("nan"))
    X, Y, meta, dates = proc.create_training_sequences(
        daily, input_days=input_days, target_days=target_days, stride=stride
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "Y.npy", Y)
    np.save(out / "meta.npy", meta)
    np.save(out / "dates.npy", dates)
    np.savez(
        out / "stats.npz",
        X_mean=X.mean(axis=(0, 1)) if len(X) else 0,
        X_std=X.std(axis=(0, 1)) if len(X) else 1,
        Y_mean=Y.mean(axis=(0, 1)) if len(Y) else 0,
        Y_std=Y.std(axis=(0, 1)) if len(Y) else 1,
    )
    typer.echo(f"Wrote {len(X)} sequences to {out}")


# ----- train -----------------------------------------------------------------


@train_app.command("simple")
def train_simple(
    epochs: int = typer.Option(50),
    batch_size: int = typer.Option(256),
    lr: float = typer.Option(1e-4),
    target_days: int = typer.Option(90),
    compile: bool = typer.Option(False, help="Use torch.compile (PyTorch 2.x)"),
    no_amp: bool = typer.Option(False),
    resume: Optional[str] = typer.Option(None),
):
    """Train SimpleLILITH on the processed GHCN data."""
    # The trainer uses argparse so we shell out to the module; cleaner than
    # restructuring the trainer to expose a Python API right now.
    import subprocess

    cmd = [
        sys.executable, "-m", "training.train_simple",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--target-days", str(target_days),
    ]
    if compile:
        cmd.append("--compile")
    if no_amp:
        cmd.append("--no-amp")
    if resume:
        cmd.extend(["--resume", resume])
    raise typer.Exit(subprocess.call(cmd))


@train_app.command("full")
def train_full(
    epochs: int = typer.Option(30),
    batch_size: int = typer.Option(4),
    variant: str = typer.Option("tiny"),
    target_days: int = typer.Option(90),
    use_grid: bool = typer.Option(False),
    compile: bool = typer.Option(False),
    no_amp: bool = typer.Option(False),
):
    """Train the full LILITH (GAT + SFNO) model. Research-novel architecture."""
    import subprocess

    cmd = [
        sys.executable, "-m", "training.train_full",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--variant", variant,
        "--target-days", str(target_days),
    ]
    if use_grid:
        cmd.append("--use-grid")
    if compile:
        cmd.append("--compile")
    if no_amp:
        cmd.append("--no-amp")
    raise typer.Exit(subprocess.call(cmd))


# ----- forecast / api --------------------------------------------------------


@app.command("evaluate")
def evaluate(
    checkpoint: Optional[str] = typer.Option(None,
                                             help="Checkpoint to evaluate. Defaults to checkpoints/lilith_best.pt."),
    training_dir: str = typer.Option("data/processed/training"),
    val_fraction: float = 0.1,
    require_beats_baselines: bool = typer.Option(False,
                                                 help="Exit non-zero if the model fails to beat persistence + climatology"),
):
    """Evaluate a checkpoint against persistence and climatology baselines.

    Reports RMSE for the model and both trivial baselines on the same
    chronological val split. The trained model should beat both — if it
    doesn't, training broke or inference is corrupted.
    """
    import json

    import numpy as np
    import torch

    from inference.baselines import (
        assert_model_beats_baselines,
        evaluate_baselines,
    )
    from inference.simple_forecaster import SimpleForecaster

    train_dir = Path(training_dir)
    if not (train_dir / "X.npy").exists():
        typer.echo(f"No training data at {train_dir}. Run `lilith process ghcn` first.", err=True)
        raise typer.Exit(1)

    ckpt = checkpoint or os.environ.get("LILITH_CHECKPOINT") or "checkpoints/lilith_best.pt"
    if not Path(ckpt).exists():
        typer.echo(f"Checkpoint not found: {ckpt}", err=True)
        raise typer.Exit(1)

    X = np.load(train_dir / "X.npy")
    Y = np.load(train_dir / "Y.npy")
    meta = np.load(train_dir / "meta.npy")
    dates_p = train_dir / "dates.npy"
    dates = np.load(dates_p) if dates_p.exists() else None
    stats = np.load(train_dir / "stats.npz")

    # Apply same normalization as the trainer
    X_n = (X - stats["X_mean"]) / (stats["X_std"] + 1e-6)
    Y_n = (Y - stats["Y_mean"]) / (stats["Y_std"] + 1e-6)
    meta_n = meta.copy()
    meta_n[:, 0] /= 90.0
    meta_n[:, 1] /= 180.0
    meta_n[:, 2] /= 5000.0

    # Chronological split
    if dates is not None:
        order = np.argsort(dates)
        cut = int(len(order) * (1 - val_fraction))
        val_idx = order[cut:]
    else:
        val_idx = np.arange(int(len(X) * (1 - val_fraction)), len(X))

    X_val = X_n[val_idx]
    Y_val = Y_n[val_idx]
    meta_val = meta_n[val_idx]
    dates_val = dates[val_idx] if dates is not None else None

    typer.echo(f"Evaluating on {len(val_idx):,} val samples...")
    baselines = evaluate_baselines(
        X_val, Y_val, meta_val, dates_val,
        Y_mean=stats["Y_mean"], Y_std=stats["Y_std"],
    )

    # Run the model
    fc = SimpleForecaster(ckpt, device="auto")
    sq = 0.0
    n_elem = 0
    for i in range(len(val_idx)):
        # Denormalize history for the forecaster (it normalizes internally)
        hist = X[val_idx[i]]
        out = fc.forecast(
            latitude=float(meta[val_idx[i], 0]),
            longitude=float(meta[val_idx[i], 1]),
            forecast_days=Y.shape[1],
            history=hist,
            elevation=float(meta[val_idx[i], 2]),
        )
        # Pull TMAX/TMIN out of the structured response. If the checkpoint
        # was trained for a shorter horizon than Y.shape[1] (e.g. ckpt
        # max_forecast=14 but Y has 90-day targets), SimpleForecaster
        # returns only N days — slice truth to match so the comparison
        # broadcasts cleanly.
        tmax = np.array([f["temperature_high"] for f in out["forecasts"]])
        tmin = np.array([f["temperature_low"] for f in out["forecasts"]])
        n_pred = len(tmax)
        truth = Y[val_idx[i], :n_pred, :2]
        pred = np.stack([tmax, tmin], axis=-1)
        sq += float(((pred - truth) ** 2).sum())
        n_elem += pred.size

    model_rmse_temp = float(np.sqrt(sq / max(n_elem, 1)))

    summary = {
        "model_rmse_temp_C": model_rmse_temp,
        "baselines": baselines,
        "n_val": int(len(val_idx)),
    }
    typer.echo(json.dumps(summary, indent=2))

    if require_beats_baselines:
        try:
            assert_model_beats_baselines(model_rmse_temp, baselines)
        except AssertionError as exc:
            typer.echo(f"\nFAIL: {exc}", err=True)
            raise typer.Exit(1)


@app.command("forecast")
def forecast(
    lat: float = typer.Option(..., help="Latitude (-90, 90)"),
    lon: float = typer.Option(..., help="Longitude (-180, 180)"),
    days: int = typer.Option(14, help="Forecast horizon"),
    checkpoint: Optional[str] = typer.Option(None,
                                             help="Path to .pt (default: checkpoints/lilith_best.pt)"),
    ensemble: int = typer.Option(0, help="MC Dropout sample count (0 = deterministic)"),
    elevation: float = typer.Option(0.0),
):
    """Run a single-location forecast against a saved checkpoint."""
    import json

    from inference.simple_forecaster import SimpleForecaster

    ckpt = checkpoint or os.environ.get("LILITH_CHECKPOINT") or "checkpoints/lilith_best.pt"
    if not Path(ckpt).exists():
        typer.echo(f"Checkpoint not found: {ckpt}", err=True)
        raise typer.Exit(1)

    fc = SimpleForecaster(ckpt, device="auto")
    out = fc.forecast(
        latitude=lat, longitude=lon, forecast_days=days,
        elevation=elevation, ensemble_samples=ensemble,
    )
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("api")
def api(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False),
    checkpoint: Optional[str] = typer.Option(None,
                                             help="Set LILITH_CHECKPOINT for the API"),
):
    """Start the FastAPI server."""
    import uvicorn

    if checkpoint:
        os.environ["LILITH_CHECKPOINT"] = checkpoint
    uvicorn.run("web.api.main:app", host=host, port=port, reload=reload)


@app.command("version")
def version():
    """Print package version."""
    try:
        from importlib.metadata import version as _v
        typer.echo(f"lilith {_v('lilith')}")
    except Exception:
        typer.echo("lilith (development)")


if __name__ == "__main__":
    app()
