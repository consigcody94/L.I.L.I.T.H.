---
language: en
tags:
- weather
- time-series
- pytorch
- climate
- transformer
- forecasting
license: apache-2.0
model-index:
- name: LILITH
  results: []
---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,25:1a1b3a,50:1e3a5f,75:3b82f6,100:0d1117&height=250&section=header&text=L.I.L.I.T.H.&fontSize=70&fontColor=ffffff&animation=fadeIn&fontAlignY=30&desc=Long-range%20Intelligent%20Learning%20for%20Integrated%20Trend%20Hindcasting&descAlignY=55&descSize=16&descColor=8b949e"/>

<br/>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Model-yellow?style=flat-square)](https://huggingface.co/consigcody94/Lilith-Weather)
[![GHCN](https://img.shields.io/badge/GHCN-100K+_Stations-22c55e?style=flat-square)](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
[![Forecast](https://img.shields.io/badge/Forecast-90_Days-0891b2?style=flat-square)]()

<br/>

**ML-powered 90-day weather forecasting trained on 150+ years of GHCN station data.**<br/>
Runs on consumer GPUs. No proprietary data. No API dependencies.

<br/>

[Quick Start](#quick-start) &ensp;&bull;&ensp; [Architecture](#architecture) &ensp;&bull;&ensp; [Training](#training) &ensp;&bull;&ensp; [Data Sources](#data-sources) &ensp;&bull;&ensp; [API](#api-reference) &ensp;&bull;&ensp; [Contributing](#contributing)

</div>

<br/>

> *"The storm goddess sees all horizons."*

<br/>

## Why L.I.L.I.T.H.

Operational weather models (GFS, ECMWF IFS) require supercomputers, petabytes of assimilated data, and institutional access. ML weather models (GraphCast, Pangu-Weather, FourCastNet) train on ERA5 reanalysis &mdash; terabytes of gridded data that most researchers cannot access or afford to process.

L.I.L.I.T.H. takes a different approach:

- **Station-native** &mdash; Learns directly from GHCN ground observations, no reanalysis required
- **Consumer hardware** &mdash; Trains on a single GPU (RTX 3060 12 GB) in hours, not days
- **90-day horizon** &mdash; Multi-scale temporal processing: synoptic (days 1&ndash;14), weekly (15&ndash;42), seasonal (43&ndash;90)
- **Uncertainty quantification** &mdash; Gaussian, quantile, or MC dropout ensemble heads
- **Full stack** &mdash; FastAPI backend + Next.js 14 frontend + Docker deployment

<br/>

## Features

| | |
|:--|:--|
| **Station-Graph Temporal Transformer** | GATv2 encoder &rarr; SFNO processor &rarr; multi-scale decoder |
| **100,000+ GHCN stations** | 150+ years of quality-controlled daily observations |
| **Climate embeddings** | ENSO, NAO, PDO, MJO, AO indices for long-range skill |
| **INT8 / INT4 quantization** | 2&ndash;4&times; memory reduction for edge deployment |
| **Keyless real-time data** | Open-Meteo integration &mdash; real observations, near-term blending, and live accuracy verification with no API key |
| **Real-time API** | FastAPI with batch inference, station lookup, and 15-minute caching |
| **Interactive frontend** | Next.js 14 with Tailwind, forecast charts, uncertainty bands |

<br/>

## Quick Start

### Install

```bash
git clone https://github.com/consigcody94/L.I.L.I.T.H..git
cd L.I.L.I.T.H.

# Core dependencies
pip install -e .

# With training extras
pip install -e ".[train]"

# With development tools
pip install -e ".[dev]"
```

### Download Data & Train

```bash
# Download GHCN station data (505 US stations, ~9.6M records)
python scripts/download_data.py --max-stations 500

# Process into training sequences
python scripts/process_data.py

# Train the model
python -m training.train_simple --epochs 50 --batch-size 64 --lr 1e-4
```

### Run Inference

```bash
# Use a trained checkpoint
python scripts/run_inference.py \
    --checkpoint checkpoints/lilith_best.pt \
    --lat 40.7128 --lon -74.006 \
    --days 90

# Start the API server
LILITH_CHECKPOINT=checkpoints/lilith_best.pt python -m uvicorn web.api.main:app --port 8000

# Query the API
curl -X POST http://localhost:8000/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"latitude": 40.7128, "longitude": -74.006, "days": 90}'
```

### Web Interface

```bash
cd web/frontend
npm install
npm run dev
# Open http://localhost:3000
```

### Docker

```bash
docker-compose -f docker/docker-compose.yml up -d
```

<br/>

## Architecture

L.I.L.I.T.H. uses a **Station-Graph Temporal Transformer (SGTT)** architecture:

```
  Station Observations (100K+ GHCN stations)
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  ENCODER                         │
  │  Station Embedding (3D + feat)   │
  │  → GATv2 (spatial correlations)  │
  │  → Temporal Transformer (RoPE)   │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  LATENT ATMOSPHERIC STATE        │
  │  64 × 128 × 256                 │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  PROCESSOR                       │
  │  SFNO (spherical harmonics)      │
  │  Multi-Scale Temporal:           │
  │    Days 1-14:  6h steps          │
  │    Days 15-42: 24h steps         │
  │    Days 43-90: 168h steps        │
  │  Climate Embedding (ENSO/NAO/..) │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  DECODER                         │
  │  Grid Decoder (global fields)    │
  │  Station Decoder (point fcsts)   │
  │  Ensemble Head (uncertainty)     │
  └──────────────────────────────────┘
```

### Model Variants

| Variant | Parameters | VRAM (FP16) | VRAM (INT8) | Use Case |
|:--|--:|--:|--:|:--|
| **LILITH-Tiny** | 50M | 4 GB | 2 GB | Edge deployment, fast inference |
| **SimpleLILITH** | 1.87M | ~23 MB | &mdash; | Default training, consumer GPUs |
| **LILITH-Base** | 150M | 8 GB | 4 GB | Balanced accuracy / speed |
| **LILITH-Large** | 400M | 12 GB | 6 GB | High-accuracy forecasts |
| **LILITH-XL** | 1B | 24 GB | 12 GB | Research, maximum accuracy |

### Key Components

| Component | Purpose | Details |
|:--|:--|:--|
| `StationEmbedding` | Encode station features + 3D position | MLP with spherical coordinates |
| `GATEncoder` | Spatial relationships | Graph Attention Network v2 |
| `TemporalTransformer` | Time series processing | Flash Attention + RoPE |
| `SFNO` | Global atmospheric dynamics | Spherical Fourier Neural Operator, O(N log N) |
| `ClimateEmbedding` | Long-range climate indices | ENSO, MJO, NAO, seasonal cycles |
| `EnsembleHead` | Uncertainty quantification | Diffusion / Gaussian / Quantile / MC dropout |
| `SimpleLILITH` | Single-station encoder-decoder | Lightweight Transformer for training |

<br/>

## Training

### Pre-trained Model

A pre-trained checkpoint is published on [HuggingFace](https://huggingface.co/consigcody94/Lilith-Weather):

- **505 US GHCN stations**, 9.6 million weather records
- **1.15 million training sequences**
- **Final RMSE: 3.88&deg;C** (temperature prediction)

```bash
# Download pre-trained checkpoint from HuggingFace
mkdir -p checkpoints && curl -L -o checkpoints/lilith_best.pt \
  https://huggingface.co/consigcody94/Lilith-Weather/resolve/main/lilith_best.pt

# Start API with trained model
LILITH_CHECKPOINT=checkpoints/lilith_best.pt python -m uvicorn web.api.main:app --port 8000
```

### Training from Scratch

```bash
# Full pipeline
python scripts/download_data.py --max-stations 500
python scripts/process_data.py
python -m training.train_simple --epochs 50 --batch-size 64

# Resume from checkpoint
python -m training.train_simple \
    --resume checkpoints/lilith_best.pt \
    --epochs 100 --lr 5e-5
```

### GPU Requirements

| GPU | Training (50 epochs, 1M samples) | Inference (single location) |
|:--|:--|:--|
| **RTX 3060 12 GB** | ~5 hours | 0.8s |
| **RTX 4090 24 GB** | ~1.5 hours | 0.3s |
| **CPU only** | ~24 hours | 3s |

### Performance Targets

| Forecast Range | Metric | L.I.L.I.T.H. Target | Climatology Baseline |
|:--|:--|--:|--:|
| Days 1&ndash;7 | Temperature RMSE | &lt; 2&deg;C | ~5&deg;C |
| Days 8&ndash;14 | Temperature RMSE | &lt; 3&deg;C | ~5&deg;C |
| Days 15&ndash;42 | Skill Score | &gt; 0.3 | 0.0 |
| Days 43&ndash;90 | Skill Score | &gt; 0.1 | 0.0 |

### Quantization

```bash
# INT8 quantization (2× memory reduction)
python inference/quantize.py --checkpoint checkpoints/lilith_best.pt --bits 8

# INT4 quantization (4× memory reduction)
python inference/quantize.py --checkpoint checkpoints/lilith_best.pt --bits 4
```

### Upload to HuggingFace

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="checkpoints/lilith_best.pt",
    path_in_repo="lilith_base_v1.pt",
    repo_id="your-username/lilith-weather",
    repo_type="model"
)
```

<br/>

## Data Sources

L.I.L.I.T.H. is built entirely on **freely available public data**.

### Primary: GHCN (Required)

| Dataset | Coverage | Stations | Variables | Resolution |
|:--|:--|--:|:--|:--|
| **GHCN-Daily** | 1763&ndash;present | 100,000+ | Temp, Precip, Snow | Daily |
| **GHCN-Hourly** | 1900s&ndash;present | 20,000+ | Wind, Pressure, Humidity | Hourly |

Source: [NOAA NCEI](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)

### Serving: Open-Meteo (keyless, no account)

At inference time L.I.L.I.T.H. uses [Open-Meteo](https://open-meteo.com) for real
data &mdash; no API key, no sign-up. It supplies recent observations to seed the model,
a real short-range forecast for near-term blending, observed values for accuracy
verification, and place-name geocoding. The client (`data/providers/open_meteo.py`)
uses only the Python standard library and fails soft (every call returns `None` on
error rather than raising).

### Recommended Supplementary Data

| Priority | Dataset | What It Adds |
|:--|:--|:--|
| High | **Climate Indices** (ENSO, NAO, MJO, PDO, AO) | Long-range predictability drivers |
| High | **ERA5 Reanalysis** (ECMWF) | Full atmospheric state, gridded global |
| Medium | **NOAA OISST** | Sea surface temperatures, ocean influence |
| Medium | **GFS Analysis** | Physics-based ensemble blending |
| Optional | **GOES/GPM Satellite** | Real-time cloud cover and precipitation |

```bash
# Download climate indices (small, fast)
python -m data.download.climate_indices --indices enso,nao,pdo,mjo,ao

# Download ERA5 for a region (requires ECMWF CDS account)
python -m data.download.era5 --start-year 2000 --end-year 2024 --region north_america
```

<br/>

## API Reference

### `POST /v1/forecast`

Generate a point forecast.

```json
{
  "latitude": 40.7128,
  "longitude": -74.006,
  "days": 90,
  "ensemble_members": 10,
  "variables": ["temperature", "precipitation", "wind"]
}
```

**Response:**

```json
{
  "location": {"latitude": 40.7128, "longitude": -74.006},
  "generated_at": "2025-01-15T12:00:00Z",
  "model_version": "SimpleLILITH v1 (RMSE: 3.88°C, source: reference+model+observations)",
  "forecast_days": 90,
  "forecasts": [
    {
      "date": "2025-01-16",
      "temperature_max": 6.8,
      "temperature_min": -1.2,
      "precipitation": 2.1,
      "precipitation_probability": 0.35,
      "temperature_max_lower": 4.6,
      "temperature_max_upper": 9.0,
      "temperature_min_lower": -3.4,
      "temperature_min_upper": 1.0
    }
  ]
}
```

### `POST /v1/forecast/batch`

Batch inference for multiple locations.

### `GET /v1/stations`

List GHCN training stations; filter by `latitude`/`longitude`/`radius` to find the
nearest stations (sorted by great-circle distance).

### `GET /v1/stations/{station_id}`

Look up a single training station by GHCN ID.

### `GET /v1/accuracy`

Prediction-accuracy report. Past predictions are verified against **real** observed
values (Open-Meteo) and aggregated into MAE/RMSE by lead time &mdash; no fabricated actuals.

### `POST /v1/forecast/hourly`

Hourly forecast (diurnal interpolation of the daily forecast).

### `GET /health`

Health check and model status.

<br/>

## Project Structure

```
L.I.L.I.T.H./
├── models/                        Model architecture
│   ├── simple_lilith.py           SimpleLILITH (shared train/inference)
│   ├── lilith.py                  Full LILITH model (SGTT)
│   ├── losses.py                  Multi-task loss functions
│   └── components/                Building blocks
│       ├── station_embed.py         Station embedding (3D + features)
│       ├── gat_encoder.py           GATv2 spatial encoder
│       ├── temporal_transformer.py  Flash Attention + RoPE
│       ├── sfno.py                  Spherical Fourier Neural Operator
│       ├── climate_embed.py         Climate index embedding
│       └── ensemble_head.py         Uncertainty quantification
│
├── training/                      Training infrastructure
│   ├── train_simple.py            SimpleLILITH training loop
│   └── trainer.py                 Full trainer with DeepSpeed
│
├── inference/                     Inference and serving
│   ├── simple_forecaster.py       Checkpoint loading + forecast generation
│   ├── forecast_blend.py          Bias correction + blending (NumPy only, testable)
│   ├── forecast.py                High-level forecast API
│   └── quantize.py                INT8/INT4 quantization
│
├── data/                          Data pipeline
│   ├── download/                  GHCN download scripts
│   ├── processing/                QC, normalization, gridding
│   ├── providers/                 Keyless real-time data (Open-Meteo)
│   └── loaders/                   PyTorch datasets
│
├── web/
│   ├── api/                       FastAPI backend (+ accuracy.py aggregation)
│   └── frontend/                  Next.js 14 frontend
│
├── scripts/                       CLI utilities
├── tests/                         Test suite
├── docker/                        Containerization
└── docs/                          Documentation
```

<br/>

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|:--|:--:|:--|:--|
| `LILITH_CHECKPOINT` | No | Auto-detected | Path to model checkpoint |
| `OPENWEATHER_API_KEY` | No | &mdash; | OpenWeatherMap key (secondary fallback only) |

**No API key is ever required.** Real recent observations and a real short-range
reference forecast are pulled from [Open-Meteo](https://open-meteo.com), which is
keyless and free for non-commercial use. This means:

- With a trained checkpoint &mdash; the model produces the forecast; real
  observations seed it and a real short-range forecast is blended into the near
  term (see [How forecasts are produced](#how-forecasts-are-produced)).
- With **no checkpoint at all** &mdash; the API still returns a real, full-horizon
  forecast (Open-Meteo near term + seasonal climatology long range). Nothing is ever
  fabricated; if no real data is reachable the API returns `503` rather than mock numbers.

`OPENWEATHER_API_KEY` is only consulted as a secondary fallback if Open-Meteo is
unreachable and a key happens to be configured.

### How forecasts are produced

The trained model's prediction is the core signal. Rather than being used raw (or,
worse, discarded), its day-to-day structure is **bias-corrected against a
seasonally-aware climatology** and combined with real data:

1. **Seed** &mdash; real recent daily observations (Open-Meteo) become the model input.
2. **Predict** &mdash; SimpleLILITH forecasts the horizon; its anomaly signal is anchored
   to per-day climatology so absolute bias is removed but real warm/cold spells survive.
3. **Blend** &mdash; for the first ~2 weeks, a real numerical-weather-prediction forecast
   is blended in (NWP dominates short-range skill), fading out smoothly.
4. **Decay** &mdash; beyond the trained 14-day window the model's weight decays toward
   climatology, where subseasonal skill is genuinely low.

Output is fully deterministic &mdash; no random perturbations. Each forecast day is
tagged with its `source` (`reference+model`, `model+climatology`, or `climatology`).

<br/>

## Acknowledgments

### Research Community

- **GraphCast** (Google DeepMind) &mdash; Pioneering ML weather prediction
- **Pangu-Weather** (Huawei) &mdash; Transformer architectures for weather
- **FourCastNet** (NVIDIA) &mdash; Fourier neural operators for atmospheric modeling
- **FuXi** (Fudan University) &mdash; Subseasonal forecasting advances

### Data Providers

- **NOAA NCEI** &mdash; GHCN dataset, a public resource funded by U.S. taxpayers
- **ECMWF** &mdash; ERA5 reanalysis data

<br/>

## Contributing

Contributions are welcome. L.I.L.I.T.H. is built on the principle that weather forecasting should be accessible to everyone.

```bash
# Development setup
git clone https://github.com/consigcody94/L.I.L.I.T.H..git
cd L.I.L.I.T.H.
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v
```

- **Code** &mdash; Model improvements, new features, bug fixes
- **Data** &mdash; Additional data sources, quality control improvements
- **Testing** &mdash; Unit tests, integration tests, benchmarking
- **Documentation** &mdash; Tutorials, guides, architecture deep-dives

<br/>

## Citation

```bibtex
@software{lilith2025,
  author = {Churchwell, Cody},
  title  = {L.I.L.I.T.H.: Long-range Intelligent Learning for Integrated Trend Hindcasting},
  year   = {2025},
  url    = {https://github.com/consigcody94/L.I.L.I.T.H.}
}
```

<br/>

## License

[Apache License 2.0](LICENSE)

<br/>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,25:1a1b3a,50:1e3a5f,75:3b82f6,100:0d1117&height=100&section=footer"/>

<sub>
Weather prediction should be free. The data is public. The science is open. Now the tools are too.
</sub>

</div>
