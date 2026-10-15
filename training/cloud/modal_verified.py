"""Bounded private Modal jobs. No endpoint or recurring schedule is deployed.

uv run --with modal modal run training/cloud/modal_verified.py --horizon 14
Download artifacts: modal volume get lilith-verified-research runs/<run-name> <local-dir>
"""

from pathlib import Path
import re
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/source")
app = modal.App("lilith-verified-research")
volume = modal.Volume.from_name("lilith-verified-research", create_if_missing=True)
cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==2.2.6")
gpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==2.2.6", "torch==2.8.0")
if modal.is_local():
    cpu_image = (cpu_image
                 .add_local_dir(ROOT / "data", "/source/data", ignore=["raw", "processed", "storage", "__pycache__"])
                 .add_local_dir(ROOT / "training", "/source/training", ignore=["__pycache__", "cloud"]))
    gpu_image = (gpu_image
                 .add_local_dir(ROOT / "models", "/source/models", ignore=["__pycache__"])
                 .add_local_dir(ROOT / "training", "/source/training", ignore=["__pycache__", "cloud"]))


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=1200,
              volumes={"/artifacts": volume}, max_containers=1, retries=0)
def prepare(horizon: int, stations: int, start_year: int, run_name: str):
    import sys
    sys.path.insert(0, "/source")
    from training.prepare_global import build_dataset
    dataset_path = Path(f"/artifacts/datasets/{run_name}")
    if (dataset_path / "manifest.json").exists():
        raise ValueError("Dataset run name already exists; choose a new immutable run name")
    manifest = build_dataset(dataset_path,
                             Path("/artifacts/cache"), limit=stations, horizon=horizon, start_year=start_year)
    volume.commit()
    return manifest["counts"]


@app.function(image=gpu_image, gpu="L4", cpu=4, memory=16384, timeout=1800,
              volumes={"/artifacts": volume}, max_containers=1, retries=0)
def fit(horizon: int, epochs: int, run_name: str):
    import sys
    sys.path.insert(0, "/source")
    from training.train_verified import train
    volume.reload()
    if Path(f"/artifacts/runs/{run_name}/report.json").exists():
        raise ValueError("Model run already exists; choose a new run name")
    report = train(Path(f"/artifacts/datasets/{run_name}"),
                   Path(f"/artifacts/runs/{run_name}"), epochs=epochs, max_seconds=1200)
    volume.commit()
    return {"run_name": run_name, "status": report["status"],
            "test_rmse": report["evaluation"]["test"]["model"]["rmse"]}


@app.local_entrypoint()
def main(horizon: int = 14, stations: int = 72, epochs: int = 12,
         run_name: str = "global-h14-1980-20261014", start_year: int = 1980):
    if horizon not in (14, 90) or not 10 <= stations <= 144 or not 1 <= epochs <= 30:
        raise ValueError("Bounded experiment: horizon 14/90; stations 10-144; epochs 1-30")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_name):
        raise ValueError("Run name must be a simple identifier")
    print(prepare.remote(horizon, stations, start_year, run_name))
    print(fit.remote(horizon, epochs, run_name))
