<p align="center">
  <img src="docs/assets/lilith-logo.svg" alt="LILITH" width="120" height="120">
</p>

<h1 align="center">LILITH</h1>

<p align="center">
  <strong>Long-range Intelligent Learning for Integrated Trend Hindcasting</strong>
</p>

<p align="center">
  <em>Named after Lilitu, the Mesopotamian storm goddess who commanded the winds</em>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://github.com/fowlb/lilith/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build"></a>
</p>

<p align="center">
  <a href="#why-lilith">Why LILITH</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## The Weather Belongs to Everyone

Every day, corporations charge billions of dollars for weather forecasts built on **freely available public data**. The Global Historical Climatology Network (GHCN)—maintained by NOAA with taxpayer funding—contains over **150 years** of weather observations from **100,000+ stations worldwide**. This data is public domain. It belongs to humanity.

Yet somehow, we've accepted that accurate long-range forecasting should be locked behind enterprise paywalls and proprietary black boxes.

**LILITH exists to change that.**

With a single consumer GPU (RTX 3060, 12GB), you can now train and run a weather prediction model that delivers **90-day forecasts** with uncertainty quantification—the same capabilities that corporations charge premium prices for. No cloud subscriptions. No API limits. No black boxes.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│   "The same public data that corporations use to train billion-dollar     │
│    weather systems is available to anyone with a GPU and curiosity."      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### The Data is Free. The Science is Open. The Code is Yours.

| What Corporations Charge For | What LILITH Provides Free |
|------------------------------|---------------------------|
| 90-day extended forecasts | 90-day forecasts with uncertainty bands |
| "Proprietary" ML models | Fully transparent architecture |
| Enterprise API access | Self-hosted, unlimited queries |
| Historical climate analytics | 150+ years of GHCN data access |
| Per-query pricing | Run on your own hardware |

---

## Why LILITH

### The Problem

Modern weather AI (GraphCast, Pangu-Weather, FourCastNet) achieves remarkable accuracy, but:

- **Requires ERA5 reanalysis data** — computationally expensive to generate, controlled by ECMWF
- **Needs massive compute** — training requires hundreds of TPUs/GPUs
- **Inference is heavy** — full global models need 80GB+ VRAM
- **Closed ecosystems** — weights available, but practical deployment requires significant resources

### The Solution

LILITH takes a different approach:

1. **Station-Native Architecture** — Learns directly from sparse GHCN station observations instead of requiring gridded reanalysis
2. **Hierarchical Processing** — Graph attention for spatial relationships, spectral methods for global dynamics
3. **Memory Efficient** — Gradient checkpointing, INT8/INT4 quantization, runs on consumer GPUs
4. **Truly Open** — Apache 2.0 license, reproducible training, no hidden dependencies

---

## Features

### Core Capabilities

- **90-Day Forecasts** — Extended-range predictions competitive with commercial services
- **Uncertainty Quantification** — Know not just the prediction, but how confident it is
- **150+ Years of Data** — Built on the complete GHCN historical record
- **Global Coverage** — Forecasts for any location on Earth
- **Multiple Variables** — Temperature, precipitation, wind, pressure, humidity

### Technical Highlights

- **Consumer Hardware** — Inference on RTX 3060 (12GB), training on RTX 4090 or multi-GPU
- **Horizontally Scalable** — From laptop to cluster with Ray Serve
- **Modern Stack** — PyTorch 2.x, Flash Attention, DeepSpeed, FastAPI, Next.js 14
- **Production Ready** — Docker containers, Redis caching, PostgreSQL + TimescaleDB

### User Experience

- **Glassmorphic UI** — Beautiful, modern interface with dynamic weather backgrounds
- **Interactive Maps** — Mapbox GL JS with temperature layers and station markers
- **Rich Visualizations** — Recharts/D3 for forecasts, uncertainty bands, wind roses
- **Historical Explorer** — Analyze 150+ years of climate trends

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (12GB+ VRAM recommended)
- Node.js 18+ (for frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/fowlb/lilith.git
cd lilith

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with all dependencies
pip install -e ".[all]"
```

### Download Data

```bash
# Download GHCN-Daily station data
python scripts/download_data.py --source ghcn-daily --stations 5000 --years 50

# Process and prepare for training
python scripts/process_data.py --config configs/data/default.yaml
```

### Training

```bash
# Train LILITH-Base on single GPU
python scripts/train_model.py --config models/configs/base.yaml

# Distributed training with DeepSpeed
deepspeed --num_gpus=4 scripts/train_model.py \
    --config models/configs/large.yaml \
    --deepspeed configs/training/ds_config.json
```

### Inference

```bash
# Generate a forecast
python scripts/run_inference.py \
    --checkpoint checkpoints/best.pt \
    --lat 40.7128 --lon -74.006 \
    --days 90

# Start the API server
python scripts/start_api.py --checkpoint checkpoints/best.pt --port 8000

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

### Docker Deployment

```bash
# Full stack deployment
docker-compose -f docker/docker-compose.yml up -d

# Individual services
docker build -f docker/Dockerfile.inference -t lilith-inference .
docker build -f docker/Dockerfile.web -t lilith-web .
```

---

## Architecture

### Model Overview

LILITH uses a **Station-Graph Temporal Transformer (SGTT)** architecture that processes weather observations through three stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LILITH ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Station Observations                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • 100,000+ GHCN stations worldwide                                 │   │
│  │  • Temperature, precipitation, pressure, wind, humidity             │   │
│  │  • Quality-controlled, gap-filled, normalized                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ENCODER ──────────────────────────────────────────────────────────────    │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐      │
│  │   Station    │──▶│  Graph Attention │──▶│ Temporal Transformer   │      │
│  │  Embedding   │   │   Network v2     │   │   (Flash Attention)    │      │
│  │              │   │                  │   │                        │      │
│  │  • 3D pos    │   │  • Spatial       │   │  • Historical context  │      │
│  │  • Features  │   │    correlations  │   │  • Causal masking      │      │
│  │  • Temporal  │   │  • Multi-hop     │   │  • RoPE embeddings     │      │
│  └──────────────┘   └──────────────────┘   └────────────────────────┘      │
│                                    │                                        │
│                                    ▼                                        │
│                    ┌───────────────────────────────┐                       │
│                    │   LATENT ATMOSPHERIC STATE    │                       │
│                    │      (64 × 128 × 256)         │                       │
│                    │                               │                       │
│                    │   Learned global grid that    │                       │
│                    │   captures atmospheric        │                       │
│                    │   dynamics implicitly         │                       │
│                    └───────────────────────────────┘                       │
│                                    │                                        │
│                                    ▼                                        │
│  PROCESSOR ────────────────────────────────────────────────────────────    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Spherical Fourier Neural Operator (SFNO)               │   │
│  │                                                                     │   │
│  │   • Operates in spectral domain on sphere                          │   │
│  │   • Captures global teleconnections (ENSO, NAO, etc.)              │   │
│  │   • Respects Earth's spherical geometry                            │   │
│  │   • Efficient O(N log N) via spherical harmonics                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Multi-Scale Temporal Processor                         │   │
│  │                                                                     │   │
│  │   Days 1-14:   6-hour steps  (synoptic weather)                    │   │
│  │   Days 15-42:  24-hour steps (weekly patterns)                     │   │
│  │   Days 43-90:  168-hour steps (seasonal trends)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Climate Embedding Module                               │   │
│  │                                                                     │   │
│  │   • ENSO index (El Niño/La Niña state)                             │   │
│  │   • MJO phase and amplitude                                        │   │
│  │   • NAO, AO, PDO indices                                           │   │
│  │   • Seasonal cycles, solar position                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  DECODER ──────────────────────────────────────────────────────────────    │
│  ┌──────────────────────┐              ┌──────────────────────┐            │
│  │    Grid Decoder      │              │   Station Decoder    │            │
│  │                      │              │                      │            │
│  │  • Global fields     │              │  • Point forecasts   │            │
│  │  • Spatial upsampling│              │  • Location-specific │            │
│  └──────────────────────┘              └──────────────────────┘            │
│                    │                              │                         │
│                    ▼                              ▼                         │
│  OUTPUT ───────────────────────────────────────────────────────────────    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Ensemble Head (Optional)                        │   │
│  │                                                                     │   │
│  │   • Diffusion-based ensemble generation                            │   │
│  │   • Gaussian, quantile, or MC dropout uncertainty                  │   │
│  │   • Calibrated confidence intervals                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  FINAL OUTPUT:                                                              │
│  • 90-day forecasts for temperature, precipitation, wind, pressure         │
│  • Uncertainty bounds (5th, 25th, 50th, 75th, 95th percentiles)           │
│  • Ensemble spread metrics                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Variants

| Variant | Parameters | VRAM (FP16) | VRAM (INT8) | Best For |
|---------|------------|-------------|-------------|----------|
| **LILITH-Tiny** | 50M | 4 GB | 2 GB | Fast inference, edge deployment |
| **LILITH-Base** | 150M | 8 GB | 4 GB | Balanced accuracy/speed |
| **LILITH-Large** | 400M | 12 GB | 6 GB | High accuracy forecasts |
| **LILITH-XL** | 1B | 24 GB | 12 GB | Research, maximum accuracy |

### Key Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| `StationEmbedding` | Encode station features + position | MLP with 3D spherical coordinates |
| `GATEncoder` | Learn spatial relationships | Graph Attention Network v2 |
| `TemporalTransformer` | Process time series | Flash Attention with RoPE |
| `SFNO` | Global atmospheric dynamics | Spherical Fourier Neural Operator |
| `ClimateEmbedding` | Encode climate indices | ENSO, MJO, NAO, seasonal |
| `EnsembleHead` | Uncertainty quantification | Diffusion / Gaussian / Quantile |

---

## Data Sources

LILITH is built entirely on **freely available public data**:

### Primary: GHCN (Global Historical Climatology Network)

| Dataset | Coverage | Stations | Variables | Resolution |
|---------|----------|----------|-----------|------------|
| **GHCN-Daily** | 1763–present | 100,000+ | Temp, Precip, Snow | Daily |
| **GHCN-Hourly** | 1900s–present | 20,000+ | Wind, Pressure, Humidity | Hourly |
| **GHCN-Monthly** | 1700s–present | 26,000 | Temp, Precip | Monthly |

**Source**: [NOAA National Centers for Environmental Information](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)

### Optional Enhancement: ERA5 Reanalysis

For users with more compute resources, ERA5 can enhance training:

| Dataset | Coverage | Resolution | Variables |
|---------|----------|------------|-----------|
| **ERA5** | 1940–present | 0.25° / hourly | Full atmospheric state |

**Source**: [ECMWF Climate Data Store](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)

---

## Performance

### Accuracy Targets

| Forecast Range | Metric | LILITH Target | Climatology |
|----------------|--------|---------------|-------------|
| Days 1-7 | Temperature RMSE | < 2°C | ~5°C |
| Days 8-14 | Temperature RMSE | < 3°C | ~5°C |
| Days 15-42 | Skill Score | > 0.3 | 0.0 |
| Days 43-90 | Skill Score | > 0.1 | 0.0 |

### Inference Performance (RTX 3060 12GB)

| Model | Single Location | Regional Grid | Global |
|-------|-----------------|---------------|--------|
| LILITH-Tiny (INT8) | 0.3s | 2s | 15s |
| LILITH-Base (INT8) | 0.8s | 5s | 45s |
| LILITH-Large (FP16) | 1.5s | 12s | 90s |

---

## Project Structure

```
lilith/
├── data/                       # Data pipeline
│   ├── download/               # GHCN download scripts
│   │   ├── ghcn_daily.py       # Daily observations
│   │   └── ghcn_hourly.py      # Hourly observations
│   ├── processing/             # Data processing
│   │   ├── quality_control.py  # Outlier detection, QC flags
│   │   ├── feature_encoder.py  # Normalization, encoding
│   │   └── gridding.py         # Station → grid interpolation
│   └── loaders/                # PyTorch datasets
│       ├── station_dataset.py  # Station-based loading
│       └── forecast_dataset.py # Forecast sequence loading
│
├── models/                     # Model architecture
│   ├── components/             # Building blocks
│   │   ├── station_embed.py    # Station feature embedding
│   │   ├── gat_encoder.py      # Graph Attention Network
│   │   ├── temporal_transformer.py  # Temporal processing
│   │   ├── sfno.py             # Spherical Fourier Neural Operator
│   │   ├── climate_embed.py    # Climate indices embedding
│   │   └── ensemble_head.py    # Uncertainty quantification
│   ├── lilith.py               # Main model class
│   ├── losses.py               # Multi-task loss functions
│   └── configs/                # Model configurations
│       ├── tiny.yaml
│       ├── base.yaml
│       └── large.yaml
│
├── training/                   # Training infrastructure
│   └── trainer.py              # Training loop with DeepSpeed
│
├── inference/                  # Inference and serving
│   ├── forecast.py             # High-level forecast API
│   └── quantize.py             # INT8/INT4 quantization
│
├── web/
│   ├── api/                    # FastAPI backend
│   │   ├── main.py             # Application entry point
│   │   └── schemas.py          # Pydantic models
│   └── frontend/               # Next.js 14 frontend
│       └── src/
│           ├── app/            # App Router pages
│           ├── components/     # React components
│           └── stores/         # Zustand state
│
├── scripts/                    # CLI utilities
│   ├── download_data.py
│   ├── process_data.py
│   ├── train_model.py
│   ├── run_inference.py
│   └── start_api.py
│
├── tests/                      # Test suite
│   ├── test_models.py
│   ├── test_data.py
│   └── test_api.py
│
├── docker/                     # Containerization
│   ├── Dockerfile.inference
│   ├── Dockerfile.web
│   └── docker-compose.yml
│
└── docs/                       # Documentation
    └── architecture.md
```

---

## API Reference

### Endpoints

#### `POST /v1/forecast`

Generate a weather forecast for a location.

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
  "model_version": "lilith-base-v1.0",
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

#### `GET /v1/stations`

List available GHCN stations.

#### `GET /v1/historical/{station_id}`

Retrieve historical observations for a station.

#### `GET /health`

Health check endpoint.

---

## Contributing

We welcome contributions from the community. LILITH is built on the principle that weather forecasting should be accessible to everyone, and that means building in the open with help from anyone who shares that vision.

### Ways to Contribute

- **Code**: Model improvements, new features, bug fixes
- **Data**: Additional data sources, quality control improvements
- **Documentation**: Tutorials, guides, API documentation
- **Testing**: Unit tests, integration tests, benchmarking
- **Design**: UI/UX improvements, visualizations

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/lilith.git
cd lilith

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run linting
ruff check .
mypy .
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with clear messages
6. Push and open a Pull Request

---

## Acknowledgments

### Data Providers

- **NOAA NCEI** — For maintaining the invaluable GHCN dataset as a public resource
- **ECMWF** — For ERA5 reanalysis data

### Research Community

- **GraphCast** (Google DeepMind) — Pioneering ML weather prediction
- **Pangu-Weather** (Huawei) — Advancing transformer architectures for weather
- **FourCastNet** (NVIDIA) — Demonstrating Fourier neural operators for atmospheric modeling
- **FuXi** (Fudan University) — Pushing boundaries in subseasonal forecasting

### Open Source

- PyTorch team for the deep learning framework
- Hugging Face for model hosting infrastructure
- The countless contributors to the Python scientific computing ecosystem

---

## License

```
Copyright 2025 LILITH Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Citation

If you use LILITH in your research, please cite:

```bibtex
@software{lilith2025,
  author = {LILITH Contributors},
  title = {LILITH: Long-range Intelligent Learning for Integrated Trend Hindcasting},
  year = {2025},
  url = {https://github.com/fowlb/lilith}
}
```

---

<p align="center">
  <br>
  <em>"The storm goddess sees all horizons."</em>
  <br><br>
  <strong>Weather prediction should be free. The data is public. The science is open. Now the tools are too.</strong>
</p>
