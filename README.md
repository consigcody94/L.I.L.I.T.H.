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

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/consigcody94/Lilith-Weather)

<br/>

[![GHCN](https://img.shields.io/badge/GHCN-100K+_Stations-22c55e?style=flat-square)](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
[![Forecast](https://img.shields.io/badge/Forecast_Horizon-90_Days-0891b2?style=flat-square)]()
[![Papers](https://img.shields.io/badge/Research_Papers-14_Implemented-a855f7?style=flat-square)]()

<br/>

### ML-powered 90-day weather forecasting trained on 150+ years of GHCN station data.

**Runs on consumer GPUs. No proprietary data. No API dependencies.**

<br/>

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Research](#-research--mathematical-foundations) · [Training](#-training) · [API](#-api-reference) · [Contributing](#-contributing)

</div>

---

<br/>

> *"The storm goddess sees all horizons."*

<br/>

## Overview

Operational weather models (GFS, ECMWF IFS) require supercomputers, petabytes of assimilated data, and institutional access. ML weather models (GraphCast, Pangu-Weather, FourCastNet) train on ERA5 reanalysis &mdash; terabytes of gridded data that most researchers cannot access or afford to process.

**L.I.L.I.T.H. takes a different approach:**

| | |
|:--|:--|
| **Station-native** | Learns directly from GHCN ground observations &mdash; no reanalysis required |
| **Consumer hardware** | Trains on a single GPU (RTX 3060 12 GB) in hours, not days |
| **90-day horizon** | Multi-scale temporal processing: synoptic (1&ndash;14d), weekly (15&ndash;42d), seasonal (43&ndash;90d) |
| **Calibrated uncertainty** | Diffusion-based ensemble, quantile regression, and MC dropout heads |
| **Physics-informed** | Hard constraints on thermodynamic consistency (NeuralGCM-inspired) |
| **Full stack** | FastAPI backend + Next.js 14 frontend + Docker deployment |

<br/>

## Highlights

<table>
<tr>
<td width="50%">

**Model Architecture**
- Station-Graph Temporal Transformer (SGTT)
- GATv2 encoder with Haversine-correct spatial graphs
- Spherical Fourier Neural Operator (SFNO) processor
- Lead-time conditioned autoregressive decoder
- Cosine-schedule diffusion ensemble head

</td>
<td width="50%">

**Training Pipeline**
- Chronological train/val split (no data leakage)
- EMA weight averaging (Polyak, decay=0.999)
- LR warmup + cosine decay schedule
- Physics consistency + extreme value losses
- Gaussian noise augmentation (std=0.02)

</td>
</tr>
<tr>
<td>

**Data & QC**
- 100,000+ GHCN stations, 150+ years
- Variable-specific spike detection thresholds
- Haversine distance for spatial operations
- IAU 2006 obliquity for solar geometry
- Mean tropical year (365.25d) cyclical encoding

</td>
<td>

**Deployment**
- INT8/INT4 quantization (2&ndash;4x memory savings)
- FastAPI with WebSocket support
- Docker Compose one-command deploy
- HuggingFace Hub integration
- Sub-second inference per location

</td>
</tr>
</table>

<br/>

## Quick Start

### Installation

```bash
git clone https://github.com/consigcody94/L.I.L.I.T.H..git
cd L.I.L.I.T.H.

pip install -e .              # Core dependencies
pip install -e ".[train]"     # + training extras
pip install -e ".[dev]"       # + development tools
```

### Train a Model

```bash
# 1. Download GHCN station data (~505 US stations, ~9.6M records)
python scripts/download_data.py --max-stations 500

# 2. Process into training sequences
python scripts/process_data.py

# 3. Train
python -m training.train_simple --epochs 50 --batch-size 64 --lr 1e-4
```

### Run Inference

```bash
# Generate a 90-day forecast
python scripts/run_inference.py \
    --checkpoint checkpoints/lilith_best.pt \
    --lat 40.7128 --lon -74.006 \
    --days 90

# Or start the API server
LILITH_CHECKPOINT=checkpoints/lilith_best.pt \
    python -m uvicorn web.api.main:app --port 8000
```

### Web Interface

```bash
cd web/frontend && npm install && npm run dev
# Open http://localhost:3000
```

### Docker (One Command)

```bash
docker-compose -f docker/docker-compose.yml up -d
```

<br/>

## Architecture

L.I.L.I.T.H. uses a **Station-Graph Temporal Transformer (SGTT)** architecture:

```
  Station Observations (100K+ GHCN stations)
                 |
                 v
  +------------------------------------------+
  |  ENCODER                                 |
  |  Station Embedding (3D spherical + feat) |
  |  --> GATv2 (spatial correlations)        |
  |  --> Temporal Transformer (Flash + RoPE) |
  +--------------------+---------------------+
                       |
                       v
  +------------------------------------------+
  |  LATENT ATMOSPHERIC STATE                |
  |  64 x 128 x 256                         |
  +--------------------+---------------------+
                       |
                       v
  +------------------------------------------+
  |  PROCESSOR                               |
  |  Spherical Fourier Neural Operator       |
  |  Multi-Scale Temporal Resolution:        |
  |    Days  1-14:  6h steps (synoptic)      |
  |    Days 15-42: 24h steps (weekly)        |
  |    Days 43-90: 168h steps (seasonal)     |
  |  Climate Embedding (ENSO/NAO/PDO/MJO)   |
  +--------------------+---------------------+
                       |
                       v
  +------------------------------------------+
  |  DECODER                                 |
  |  Lead-Time Conditioned (Stormer)         |
  |  Tendency Clamping +/-5.0 (NeuralGCM)   |
  |  Ensemble Head (cosine diffusion)        |
  +------------------------------------------+
```

### Model Variants

| Variant | Parameters | VRAM (FP16) | VRAM (INT8) | Use Case |
|:--|--:|--:|--:|:--|
| **SimpleLILITH** | 1.87M | ~23 MB | &mdash; | Default training, consumer GPUs |
| **LILITH-Tiny** | 50M | 4 GB | 2 GB | Edge deployment, fast inference |
| **LILITH-Base** | 150M | 8 GB | 4 GB | Balanced accuracy / speed |
| **LILITH-Large** | 400M | 12 GB | 6 GB | High-accuracy forecasts |
| **LILITH-XL** | 1B | 24 GB | 12 GB | Research, maximum accuracy |

### Component Overview

| Component | Module | Description |
|:--|:--|:--|
| Station Embedding | `station_embed.py` | 3D spherical coordinates + feature MLP, mean tropical year encoding |
| GATv2 Encoder | `gat_encoder.py` | Graph attention with Haversine-correct spatial edges |
| Temporal Transformer | `temporal_transformer.py` | Flash Attention + Rotary Position Embedding |
| SFNO | `sfno.py` | Spherical Fourier Neural Operator, Xavier-like spectral init, O(N log N) |
| Climate Embedding | `climate_embed.py` | ENSO, MJO, NAO, seasonal cycles, IAU 2006 solar declination |
| Ensemble Head | `ensemble_head.py` | Cosine-schedule diffusion / Gaussian / Quantile / MC dropout |
| Forecast Decoder | `lilith.py` | Lead-time embedding + tendency-clamped autoregressive rollout |

<br/>

## Research & Mathematical Foundations

L.I.L.I.T.H. incorporates techniques from **14 published research papers** spanning ML weather prediction, probabilistic forecasting, and numerical methods. All implementations include proper citations and have been verified against the original papers.

### Numerical Correctness

These fixes address silent mathematical errors that degrade forecast accuracy:

| Category | Fix | Impact |
|:--|:--|:--|
| **Geodesic math** | Haversine formula for all spatial distances | Eliminates up to 2x longitude error at 60&deg;N (1&deg; lon = 55 km, not 111 km) |
| **Solar geometry** | IAU 2006 obliquity (23.4393&deg;) | Corrects seasonal cycle amplitude by 0.3% |
| **Temporal encoding** | Mean tropical year (365.25d) divisor | Prevents Dec 31 phase discontinuity across leap years |
| **Monthly encoding** | 30.4375d (365.25/12) cycle length | Eliminates month-boundary artifacts |
| **Data leakage** | Chronological train/val split | Removes optimistic bias from adjacent-day leakage |
| **Metric computation** | Per-feature RMSE denormalization | Corrects misleading evaluation when feature scales differ |
| **QC thresholds** | Variable-specific spike detection | Prevents false flagging of normal stable-weather variation |
| **Pressure bounds** | 850 hPa lower limit | Supports high-altitude stations (e.g., La Paz, Lhasa) |

### Loss Functions

Multi-objective training with physics-informed constraints:

| Loss | Weight | Paper | Purpose |
|:--|:--:|:--|:--|
| **Weighted MSE** | 1.0 | &mdash; | Primary reconstruction loss |
| **Fair CRPS** | 0.1 | Ferro (2014) | Bias-corrected ensemble calibration via n/(n-1) correction |
| **Energy Score** | &mdash; | Gneiting & Raftery (2007) | Multivariate CRPS: joint calibration across all variables |
| **Physics Consistency** | 0.5 | NeuralGCM (Nature, 2024) | Penalizes T_max < T_min inversions, negative precipitation |
| **Extreme Value** | 0.2 | FuXi-Extreme (2024) | Upweights events beyond 2&sigma; of climatological distribution |
| **Spectral Energy** | 0.05 | GraphCast / NeuralGCM | Prevents unphysical energy generation in Fourier domain |
| **Huber Quantile** | &mdash; | Dabney et al. (2018) | Robust tail calibration, less sensitive to outliers |

### Training Techniques

| Technique | Details | Paper |
|:--|:--|:--|
| **EMA averaging** | Decay = 0.999, shadow weights for validation | Polyak (1992), Izmailov et al. (2018) |
| **LR schedule** | Linear warmup (10% epochs) + cosine decay | Vaswani et al. (2017) |
| **Weight decay** | 0.05 (aligned with Pangu-Weather) | Loshchilov & Hutter (2019) |
| **Gradient optimization** | `set_to_none=True`, PyTorch 2.1+ AMP | PyTorch best practices |
| **Data augmentation** | Gaussian noise injection (std=0.02) | Wen et al. (2020) |
| **Curriculum learning** | Stages [7, 14, 30, 60, 90] days | Smooth difficulty transitions |

### Decoder & Ensemble Innovations

| Innovation | Details | Paper |
|:--|:--|:--|
| **Lead-time embedding** | Learned per-step embedding in decoder | Stormer (Nguyen et al., ICML 2024) |
| **Tendency clamping** | &plusmn;5.0 clamp on predicted state changes | NeuralGCM (Kochkov et al., Nature 2024) |
| **Cosine noise schedule** | Replaces linear &beta; schedule in diffusion | GenCast (Price et al., Nature 2024), Nichol & Dhariwal (2021) |
| **Xavier spectral init** | 1/&radic;(in &times; out) for Fourier weights | Variance preservation in spectral domain |
| **Asymmetric diurnal curve** | Parton-Logan model for hourly temperatures | Parton & Logan (1981) |

<br/>

## Training

### Pre-trained Checkpoint

A pre-trained checkpoint is available via [GitHub Releases](https://github.com/consigcody94/L.I.L.I.T.H./releases):

| Stat | Value |
|:--|:--|
| **Stations** | 505 US GHCN stations |
| **Records** | 9.6 million weather observations |
| **Sequences** | 1.15 million training sequences |
| **Temperature RMSE** | 3.88&deg;C |

```bash
# Download and serve
curl -L -o checkpoints/lilith_best.pt \
  https://github.com/consigcody94/L.I.L.I.T.H./releases/download/v1.0/lilith_best.pt

LILITH_CHECKPOINT=checkpoints/lilith_best.pt \
    python -m uvicorn web.api.main:app --port 8000
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

### Hardware Requirements

| GPU | Training (50 epochs, 1M samples) | Inference (single location) |
|:--|--:|--:|
| **RTX 3060 12 GB** | ~5 hours | 0.8s |
| **RTX 4090 24 GB** | ~1.5 hours | 0.3s |
| **CPU only** | ~24 hours | 3s |

### Performance Targets

| Forecast Range | Metric | L.I.L.I.T.H. Target | Climatology Baseline |
|:--|:--|--:|--:|
| Days 1&ndash;7 | Temperature RMSE | &lt; 2&deg;C | ~5&deg;C |
| Days 8&ndash;14 | Temperature RMSE | &lt; 3&deg;C | ~5&deg;C |
| Days 15&ndash;42 | Skill Score (CRPSS) | &gt; 0.3 | 0.0 |
| Days 43&ndash;90 | Skill Score (CRPSS) | &gt; 0.1 | 0.0 |

### Quantization

```bash
python inference/quantize.py --checkpoint checkpoints/lilith_best.pt --bits 8   # 2x memory reduction
python inference/quantize.py --checkpoint checkpoints/lilith_best.pt --bits 4   # 4x memory reduction
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

### Supplementary Data

| Priority | Dataset | Purpose |
|:--|:--|:--|
| **High** | Climate Indices (ENSO, NAO, MJO, PDO, AO) | Long-range teleconnection drivers |
| **High** | ERA5 Reanalysis (ECMWF) | Full atmospheric state, gridded fields |
| **Medium** | NOAA OISST | Sea surface temperatures, ocean coupling |
| **Medium** | GFS Analysis | Physics-based ensemble blending |
| **Optional** | GOES/GPM Satellite | Real-time cloud cover and precipitation |

```bash
# Climate indices (small, fast download)
python -m data.download.climate_indices --indices enso,nao,pdo,mjo,ao

# ERA5 reanalysis (requires ECMWF CDS account)
python -m data.download.era5 --start-year 2000 --end-year 2024 --region north_america
```

<br/>

## API Reference

### `POST /v1/forecast`

Generate a point forecast with uncertainty quantification.

**Request:**

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
  "location": {"latitude": 40.7128, "longitude": -74.006, "name": "New York, NY"},
  "generated_at": "2025-01-15T12:00:00Z",
  "model_version": "SimpleLILITH v1",
  "forecasts": [
    {
      "date": "2025-01-16",
      "temperature": {"mean": 2.5, "min": -1.2, "max": 6.8},
      "precipitation": {"probability": 0.35, "amount_mm": 2.1},
      "wind": {"speed_ms": 5.2, "direction_deg": 270},
      "uncertainty": {"temperature_std": 1.2, "confidence": 0.85}
    }
  ]
}
```

| Endpoint | Method | Description |
|:--|:--|:--|
| `/v1/forecast` | POST | Single-location forecast |
| `/v1/forecast/batch` | POST | Multi-location batch inference |
| `/v1/historical/{station_id}` | GET | Historical observations for a GHCN station |
| `/health` | GET | Health check and model status |

<br/>

## Project Structure

```
L.I.L.I.T.H./
|
|-- models/                          Model architecture
|   |-- simple_lilith.py               SimpleLILITH (lightweight transformer)
|   |-- lilith.py                      Full SGTT model
|   |-- losses.py                      Multi-objective loss functions
|   +-- components/
|       |-- station_embed.py             Station embedding (3D + features)
|       |-- gat_encoder.py              GATv2 spatial encoder
|       |-- temporal_transformer.py      Flash Attention + RoPE
|       |-- sfno.py                      Spherical Fourier Neural Operator
|       |-- climate_embed.py             Climate index embedding
|       +-- ensemble_head.py             Uncertainty quantification
|
|-- training/                        Training infrastructure
|   |-- train_simple.py                SimpleLILITH training loop
|   +-- trainer.py                     Full trainer with DeepSpeed + EMA
|
|-- inference/                       Inference and serving
|   |-- simple_forecaster.py           Forecasting with Parton-Logan diurnal model
|   |-- forecast.py                    High-level forecast API
|   +-- quantize.py                    INT8/INT4 quantization
|
|-- data/                            Data pipeline
|   |-- download/                      GHCN download scripts
|   |-- processing/                    QC, normalization, Haversine gridding
|   +-- loaders/                       PyTorch datasets with augmentation
|
|-- web/
|   |-- api/                           FastAPI backend
|   +-- frontend/                      Next.js 14 frontend
|
|-- scripts/                         CLI utilities
|-- tests/                           Test suite
|-- docker/                          Containerization
+-- docs/                            Documentation
```

<br/>

## Configuration

| Variable | Required | Default | Description |
|:--|:--:|:--|:--|
| `LILITH_CHECKPOINT` | No | Auto-detected | Path to model checkpoint |
| `OPENWEATHER_API_KEY` | No | &mdash; | OpenWeatherMap key (fallback only) |

The ML model works without any API keys. OpenWeatherMap is only used as a fallback when no trained model is loaded.

<br/>

## Acknowledgments

### Research Papers Implemented

| Paper | Year | Contribution to L.I.L.I.T.H. |
|:--|:--:|:--|
| Polyak & Juditsky | 1992 | EMA weight averaging |
| Gneiting & Raftery | 2007 | Energy Score for multivariate evaluation |
| Parton & Logan | 1981 | Asymmetric diurnal temperature curve |
| Vaswani et al. | 2017 | Learning rate warmup schedule |
| Dabney et al. | 2018 | Huber quantile loss |
| Izmailov et al. | 2018 | Stochastic Weight Averaging |
| Brody et al. | 2021 | GATv2 attention mechanism |
| Nichol & Dhariwal | 2021 | Cosine noise schedule for diffusion |
| Ferro | 2014 | Fair CRPS for finite ensembles |
| Kochkov et al. (NeuralGCM) | 2024 | Physics constraints, tendency clamping |
| Price et al. (GenCast) | 2024 | Diffusion-based ensemble forecasting |
| Nguyen et al. (Stormer) | 2024 | Lead-time conditioned forecasting |
| Chen et al. (FuXi-Extreme) | 2024 | Extreme value loss weighting |
| IAU 2006 Resolution | 2006 | Obliquity of the ecliptic (23.4393&deg;) |

### ML Weather Prediction Community

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
git clone https://github.com/consigcody94/L.I.L.I.T.H..git
cd L.I.L.I.T.H.
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v
```

| Area | Examples |
|:--|:--|
| **Code** | Model improvements, new ensemble heads, optimizer experiments |
| **Data** | Additional data sources, QC pipeline improvements |
| **Testing** | Unit tests, integration tests, forecast benchmarking |
| **Documentation** | Tutorials, architecture deep-dives, deployment guides |

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
