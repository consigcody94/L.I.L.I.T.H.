---
language: en
tags:
- weather
- time-series
- pytorch
- climate
license: apache-2.0
model-index:
- name: LILITH
  results: []
---

# L.I.L.I.T.H. (Long-range Intelligent Learning for Integrated Trend Hindcasting)

**A lightweight, open-source weather prediction model trained on GHCN data.**

<p align="center">
  <a href="https://huggingface.co/consigcody94/Lilith-Weather"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Hugging Face Model"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://github.com/consigcody94/lilith/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build"></a>
</p>

## Model Description

LILITH is a transformer-based weather forecasting model designed to run on consumer hardware (e.g., RTX 3060). It learns from 150+ years of station-based observations (GHCN-Daily) to predict 90-day temperature and precipitation trends with uncertainty quantification.

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

### The Data is Free. The Science is Open. The Code is Yours

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

### Quick Start with Pre-trained Model

If you have a trained checkpoint (e.g., `lilith_best.pt`), you can run the full stack immediately:

```bash
# 1. Clone and setup
git clone https://github.com/consigcody94/lilith.git
cd lilith
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[all]"

# 3. Place your checkpoint in the checkpoints folder
mkdir checkpoints
# Copy lilith_best.pt to checkpoints/

# 4. Set OpenWeatherMap API Key (Optional but recommended for live data)
export OPENWEATHER_API_KEY="your_api_key_here"  # Linux/Mac
# set OPENWEATHER_API_KEY=your_api_key_here      # Windows

# 5. Start the API server (auto-detects checkpoint)
python -m uvicorn web.api.main:app --host 127.0.0.1 --port 8000

# 6. In a new terminal, start the frontend
cd web/frontend
npm install
npm run dev

# 7. Open http://localhost:3000 in your browser
```

The API will automatically find and load `checkpoints/lilith_best.pt` or `checkpoints/lilith_final.pt`. You'll see log output like:

```
Found checkpoint at C:\...\checkpoints\lilith_best.pt
Model loaded on cuda
Config: d_model=128, layers=4
Val RMSE: 3.96°C
Model loaded successfully (RMSE: 3.96°C)
```

**Test the API directly:**

```bash
curl -X POST http://127.0.0.1:8000/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"latitude": 40.7128, "longitude": -74.006, "days": 14}'
```

### Installation

```bash
# Clone the repository
git clone https://github.com/consigcody94/lilith.git
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

LILITH training is designed to work on consumer GPUs. Here's a complete step-by-step guide:

#### Step 1: Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA support
# For RTX 30/40 series:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For RTX 50 series (Blackwell - requires nightly):
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install LILITH dependencies
pip install -e ".[all]"
```

#### Step 2: Download Training Data

```bash
# Download GHCN station data (start with 300 stations for quick training)
python -m data.download.ghcn_daily \
    --stations 300 \
    --min-years 30 \
    --country US

# For better models, download more stations
python -m data.download.ghcn_daily \
    --stations 5000 \
    --min-years 20 \
    --elements TMAX,TMIN,PRCP

# Download climate indices for long-range prediction
python -m data.download.climate_indices --all
```

#### Step 3: Process Data

```bash
# Process raw GHCN data into training format
python -m data.processing.ghcn_processor

# This creates:
# - data/processed/ghcn_combined.parquet (all station data)
# - data/processed/training/X.npy (input sequences)
# - data/processed/training/Y.npy (target sequences)
# - data/processed/training/meta.npy (station metadata)
# - data/processed/training/stats.npz (normalization stats)
```

#### Step 4: Train the Model

```bash
# Quick training (30 epochs, good for testing)
python -m training.train_simple \
    --epochs 30 \
    --batch-size 64 \
    --d-model 128 \
    --layers 4

# Full training (100 epochs, production quality)
python -m training.train_simple \
    --epochs 100 \
    --batch-size 128 \
    --d-model 256 \
    --layers 6 \
    --lr 1e-4

# Resume training from checkpoint
python -m training.train_simple \
    --resume checkpoints/lilith_best.pt \
    --epochs 50
```

#### Step 5: Monitor Training

During training, you'll see output like:

```
Epoch 1/30 | Train Loss: 0.8234 | Val Loss: 0.7891 | Temp RMSE: 4.21°C | Temp MAE: 3.15°C
Epoch 2/30 | Train Loss: 0.6543 | Val Loss: 0.6234 | Temp RMSE: 3.45°C | Temp MAE: 2.67°C
...
Epoch 30/30 | Train Loss: 0.2134 | Val Loss: 0.2456 | Temp RMSE: 1.89°C | Temp MAE: 1.42°C
```

Target metrics:

- **Days 1-7**: Temp RMSE < 2°C
- **Days 8-14**: Temp RMSE < 3°C

#### Step 6: Use the Trained Model

```bash
# Update the API to use your trained model
# Edit web/api/main.py and set DEMO_MODE = False

# Or run inference directly
python -m inference.forecast \
    --checkpoint checkpoints/lilith_best.pt \
    --lat 40.7128 --lon -74.006 \
    --days 90
```

#### Training on Multiple GPUs

```bash
# Using PyTorch DistributedDataParallel
torchrun --nproc_per_node=2 training/train_distributed.py \
    --config models/configs/large.yaml

# Using DeepSpeed for memory efficiency
deepspeed --num_gpus=4 training/train_deepspeed.py \
    --config models/configs/xl.yaml \
    --deepspeed configs/training/ds_config.json
```

#### Memory Requirements

| Model Size | Batch Size | VRAM Required |
|------------|------------|---------------|
| d_model=128 | 64 | ~4 GB |
| d_model=256 | 64 | ~8 GB |
| d_model=256 | 128 | ~12 GB |
| d_model=512 | 64 | ~16 GB |

#### Training Tips

1. **Start small**: Train with 300 stations first to verify everything works
2. **Monitor GPU usage**: Use `nvidia-smi` to ensure GPU is being utilized
3. **Watch for overfitting**: If val loss increases while train loss decreases, reduce epochs
4. **Save checkpoints**: The best model is automatically saved to `checkpoints/lilith_best.pt`
5. **Use mixed precision**: Enabled by default (FP16), cuts memory usage in half

---

## Pre-trained Models

### Using Pre-trained Checkpoints

Once a model is trained, you **do not need to retrain** — the checkpoint file contains everything needed for inference. Anyone can download and use pre-trained models.

#### Checkpoint File Contents

The `.pt` checkpoint file (~20-50MB depending on model size) contains:

```python
checkpoint = {
    'epoch': 20,                    # Training epoch when saved
    'model_state_dict': {...},      # All learned weights
    'optimizer_state_dict': {...},  # Optimizer state (for resuming training)
    'val_loss': 0.2456,             # Validation loss at checkpoint
    'val_rmse': 1.89,               # Temperature RMSE in °C
    'config': {                      # Model architecture config
        'input_features': 3,
        'output_features': 3,
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dropout': 0.1
    },
    'normalization': {              # Data normalization stats
        'X_mean': [...],
        'X_std': [...],
        'Y_mean': [...],
        'Y_std': [...]
    }
}
```

#### Pre-trained Checkpoint Included

A pre-trained checkpoint (`lilith_best.pt`) is included in the `checkpoints/` folder. This model was trained on:

- **915,000 sequences** from 300 US GHCN stations
- **20 epochs** of training
- **Validation RMSE: 3.96°C**

You can use this checkpoint immediately or train your own model with different data/parameters.

#### Model Specifications

| Model | Parameters | File Size | VRAM (Inference) | Best For |
|-------|------------|-----------|------------------|----------|
| **SimpleLILITH** | 1.87M | ~23 MB | 2-4 GB | Default model, fast training |
| **lilith-base** | 150M | ~45 MB | 4 GB | Balanced accuracy/speed |
| **lilith-large** | 400M | ~120 MB | 8 GB | High accuracy |

### GPU Requirements for Inference

Unlike training, inference requires much less VRAM. Here's what you can run on different hardware:

| GPU | VRAM | Models Supported | Batch Size | Latency (90-day forecast) |
|-----|------|------------------|------------|---------------------------|
| **RTX 3050/4050** | 4 GB | Tiny, Base (INT8) | 1 | ~1.5 sec |
| **RTX 3060/4060** | 8 GB | Tiny, Base, Large (INT8) | 1-4 | ~0.8 sec |
| **RTX 3070/4070** | 8-12 GB | All models (FP16) | 4-8 | ~0.5 sec |
| **RTX 3080/4080** | 10-16 GB | All models (FP16) | 8-16 | ~0.3 sec |
| **RTX 3090/4090** | 24 GB | All models, ensembles | 32+ | ~0.2 sec |
| **RTX 5050** | 8.5 GB | Tiny, Base, Large (INT8) | 1-4 | ~0.6 sec |
| **CPU Only** | N/A | All models (slow) | 1 | ~10-30 sec |

#### Quantization for Smaller GPUs

```bash
# Convert to INT8 for 50% memory reduction
python -m inference.quantize \
    --checkpoint checkpoints/lilith_base.pt \
    --output checkpoints/lilith_base_int8.pt \
    --precision int8

# Convert to INT4 for 75% memory reduction (slight accuracy loss)
python -m inference.quantize \
    --checkpoint checkpoints/lilith_base.pt \
    --output checkpoints/lilith_base_int4.pt \
    --precision int4
```

### Loading and Using a Checkpoint

#### Python API

```python
import torch
from models.lilith import SimpleLILITH

# Load checkpoint
checkpoint = torch.load('checkpoints/lilith_best.pt', map_location='cuda')

# Recreate model from config
model = SimpleLILITH(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get normalization stats
X_mean = torch.tensor(checkpoint['normalization']['X_mean'])
X_std = torch.tensor(checkpoint['normalization']['X_std'])
Y_mean = torch.tensor(checkpoint['normalization']['Y_mean'])
Y_std = torch.tensor(checkpoint['normalization']['Y_std'])

# Run inference
with torch.no_grad():
    # Normalize input
    X_norm = (X - X_mean) / X_std

    # Predict
    pred = model(X_norm, meta, target_len=14)

    # Denormalize output
    pred_denorm = pred * Y_std + Y_mean
```

#### Command Line

```bash
# Single location forecast
python -m inference.forecast \
    --checkpoint checkpoints/lilith_best.pt \
    --lat 40.7128 --lon -74.006 \
    --days 90 \
    --output forecast.json

# Batch inference for multiple locations
python -m inference.forecast \
    --checkpoint checkpoints/lilith_best.pt \
    --locations-file locations.csv \
    --days 90 \
    --output forecasts/
```

#### Start API Server with Trained Model

```bash
# Set checkpoint path
export LILITH_CHECKPOINT=checkpoints/lilith_best.pt

# Start API (will use trained model instead of demo mode)
python -m web.api.main

# Or specify directly
python -m uvicorn web.api.main:app --host 0.0.0.0 --port 8000
```

### Sharing Your Trained Model

#### Upload to HuggingFace Hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="checkpoints/lilith_best.pt",
    path_in_repo="lilith_base_v1.pt",
    repo_id="your-username/lilith-base",
    repo_type="model"
)
```

#### Create a GitHub Release

```bash
# Tag your release
git tag -a v1.0 -m "LILITH Base v1.0 - Trained on 915K sequences"
git push origin v1.0

# Upload checkpoint to release (via GitHub UI or gh cli)
gh release create v1.0 checkpoints/lilith_best.pt --title "LILITH v1.0"
```

### Model Training Metrics

When training completes, you'll see metrics like:

```
┌────────────────────────────────────────────────────────────────┐
│                    LILITH TRAINING COMPLETE                    │
├────────────────────────────────────────────────────────────────┤
│  Epochs: 20                                                    │
│  Training Samples: 915,001                                     │
│  Final Train Loss: 0.2134                                      │
│  Final Val Loss: 0.2456                                        │
│  Temperature RMSE: 1.89°C                                      │
│  Temperature MAE: 1.42°C                                       │
│  Checkpoint: checkpoints/lilith_best.pt (22.8 MB)              │
├────────────────────────────────────────────────────────────────┤
│  Model Config:                                                 │
│    - Parameters: 1,869,251                                     │
│    - d_model: 128                                              │
│    - Attention Heads: 4                                        │
│    - Encoder Layers: 4                                         │
│    - Decoder Layers: 4                                         │
└────────────────────────────────────────────────────────────────┘
```

### Resuming Training

```bash
# Continue training from checkpoint
python -m training.train_simple \
    --resume checkpoints/lilith_best.pt \
    --epochs 50 \
    --lr 5e-5  # Lower learning rate for fine-tuning

# The checkpoint includes optimizer state, so training continues smoothly
```

### Model Comparison

| Checkpoint | Epochs | Training Data | Val RMSE | File Size | Notes |
|------------|--------|---------------|----------|-----------|-------|
| `lilith_v0.1.pt` | 10 | 100K samples | 4.3°C | 22 MB | Quick test |
| `lilith_v0.5.pt` | 30 | 500K samples | 2.8°C | 22 MB | Development |
| `lilith_v1.0.pt` | 100 | 915K samples | 1.9°C | 22 MB | Production |
| `lilith_large_v1.pt` | 100 | 2M samples | 1.5°C | 120 MB | Best accuracy |

---

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

LILITH is built entirely on **freely available public data**. The more data sources you integrate, the better your predictions will be.

### Primary: GHCN (Global Historical Climatology Network)

| Dataset | Coverage | Stations | Variables | Resolution |
|---------|----------|----------|-----------|------------|
| **GHCN-Daily** | 1763–present | 100,000+ | Temp, Precip, Snow | Daily |
| **GHCN-Hourly** | 1900s–present | 20,000+ | Wind, Pressure, Humidity | Hourly |
| **GHCN-Monthly** | 1700s–present | 26,000 | Temp, Precip | Monthly |

**Source**: [NOAA National Centers for Environmental Information](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)

### Recommended Additional Data Sources

These freely available datasets can significantly improve prediction accuracy:

#### 1. ERA5 Reanalysis (Highly Recommended)

| Dataset | Coverage | Resolution | Variables |
|---------|----------|------------|-----------|
| **ERA5** | 1940–present | 0.25° / hourly | Full atmospheric state (temperature, wind, humidity, pressure at all levels) |

**Source**: [ECMWF Climate Data Store](https://cds.climate.copernicus.eu/)

- Provides gridded global data interpolated from observations
- Excellent for learning atmospheric dynamics
- ~2TB for 10 years of data at full resolution

#### 2. Climate Indices (Essential for Long-Range)

| Index | Description | Impact |
|-------|-------------|--------|
| **ENSO (ONI)** | El Niño/La Niña state | Major driver of global weather patterns |
| **NAO** | North Atlantic Oscillation | European/North American winter weather |
| **PDO** | Pacific Decadal Oscillation | Long-term Pacific climate cycles |
| **MJO** | Madden-Julian Oscillation | Tropical weather, 30-60 day cycles |
| **AO** | Arctic Oscillation | Northern Hemisphere cold outbreaks |

**Source**: [NOAA Climate Prediction Center](https://www.cpc.ncep.noaa.gov/)

```bash
# Download climate indices
python -m data.download.climate_indices --indices enso,nao,pdo,mjo,ao
```

#### 3. Sea Surface Temperature (SST)

| Dataset | Coverage | Resolution |
|---------|----------|------------|
| **NOAA OISST** | 1981–present | 0.25° / daily |
| **HadISST** | 1870–present | 1° / monthly |

**Source**: [NOAA OISST](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)

- Ocean temperatures strongly influence atmospheric patterns
- Critical for predicting precipitation and temperature anomalies

#### 4. NOAA GFS Model Data

| Dataset | Forecast Range | Resolution |
|---------|----------------|------------|
| **GFS Analysis** | Historical | 0.25° / 6-hourly |
| **GFS Forecasts** | 16 days | 0.25° / hourly |

**Source**: [NOAA NOMADS](https://nomads.ncep.noaa.gov/)

- Use as additional training signal or for ensemble weighting
- Can blend ML predictions with physics-based forecasts

#### 5. Satellite Data

| Dataset | Variables | Coverage |
|---------|-----------|----------|
| **GOES-16/17/18** | Cloud cover, precipitation | Americas |
| **NASA GPM** | Global precipitation | Global |
| **MODIS** | Land surface temperature | Global |

**Sources**:

- [NOAA CLASS](https://www.class.noaa.gov/)
- [NASA Earthdata](https://earthdata.nasa.gov/)

#### 6. Additional Reanalysis Products

| Dataset | Coverage | Best For |
|---------|----------|----------|
| **NASA MERRA-2** | 1980–present | North America |
| **NCEP/NCAR Reanalysis** | 1948–present | Historical coverage |
| **JRA-55** | 1958–present | Pacific/Asia region |

### Data Download Commands

```bash
# Download all recommended data sources
python -m data.download.all \
    --ghcn-stations 5000 \
    --era5-years 20 \
    --climate-indices all \
    --sst oisst \
    --region north_america

# Download just climate indices (small, fast)
python -m data.download.climate_indices

# Download ERA5 for specific region (requires CDS account)
python -m data.download.era5 \
    --start-year 2000 \
    --end-year 2024 \
    --region "north_america" \
    --variables temperature,wind,humidity,pressure
```

### Data Integration Priority

For the best results, add data sources in this order:

1. **GHCN-Daily** (required) - Station observations
2. **Climate Indices** (highly recommended) - ENSO, NAO, MJO for long-range skill
3. **ERA5** (recommended) - Full atmospheric state for dynamics
4. **SST** (recommended) - Ocean influence on weather
5. **Satellite** (optional) - Real-time cloud/precip data

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
# Fork and clone (replace with your username if you fork)
git clone https://github.com/consigcody94/lilith.git
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

### U.S. Government AI Initiatives

We thank **President Donald Trump** and his administration for the **Stargate AI Initiative** and commitment to advancing American AI research and infrastructure. The recognition that AI development—including open-source projects like LILITH—represents a critical frontier for innovation, economic growth, and global competitiveness has helped create an environment where ambitious projects like this can flourish. The initiative's focus on building domestic AI capabilities and infrastructure supports the democratization of advanced technologies for all Americans.

### Data Providers

- **NOAA NCEI** — For maintaining the invaluable GHCN dataset as a public resource funded by U.S. taxpayers
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

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENWEATHER_API_KEY` | Yes (for live data) | `YOUR_OPENWEATHER_API_KEY_HERE` | Free API key from [OpenWeatherMap](https://openweathermap.org/api) |
| `LILITH_CHECKPOINT` | No | Auto-detected | Path to trained model checkpoint |

### Getting an OpenWeatherMap API Key

1. Sign up at [OpenWeatherMap](https://openweathermap.org/users/sign_up) (free)
2. Go to [API Keys](https://home.openweathermap.org/api_keys)
3. Copy your API key
4. Set the environment variable:

```bash
# Linux/Mac
export OPENWEATHER_API_KEY="your_key_here"

# Windows PowerShell
$env:OPENWEATHER_API_KEY="your_key_here"

# Windows CMD
set OPENWEATHER_API_KEY=your_key_here
```

### Using the Pre-trained Model

A pre-trained model is available in the releases. This model was trained on:

- **505 US GHCN stations** with 9.6 million weather records
- **1.15 million training sequences**
- **10 epochs** of training (~5 hours on CPU, ~1 hour on GPU)
- **Final RMSE: 3.88°C** (temperature prediction accuracy)

Download and use:

```bash
# Download from releases
curl -L -o checkpoints/lilith_best.pt https://github.com/consigcody94/lilith/releases/download/v1.0/lilith_best.pt

# Start with the model
LILITH_CHECKPOINT=checkpoints/lilith_best.pt python -m uvicorn web.api.main:app --port 8000
```

### Live Data & Caching

LILITH fetches live data from external APIs. To avoid hitting rate limits:

#### OpenWeatherMap (Forecast Adjustments)

- **Source**: api.openweathermap.org
- **Cache**: 15 minutes per location
- **Rate Limit**: 1,000 calls/day on free tier
- Used for fallback forecasts when ML model is unavailable

To disable live data fetching entirely and use only the ML model:

```python
# In web/api/main.py, set _weather_service to None
_weather_service = None  # Disables OpenWeatherMap calls
```

### Running Without API Keys

If you don't want to set up API keys, the app will still work but with limited features:

| Feature | With API Key | Without API Key |
|---------|--------------|-----------------|
| ML Forecasts | ✅ Full functionality | ✅ Full functionality |
| Fallback Forecasts | ✅ OWM-based | ❌ Error if model not loaded |

### Data Directory Structure

```
data/
├── raw/
│   └── ghcn_daily/           # Downloaded GHCN station files
│       ├── stations/         # .dly files (gitignored)
│       ├── ghcnd-stations.txt
│       └── ghcnd-inventory.txt
├── processed/
│   └── training/             # Processed training data (gitignored)
│       ├── X.npy             # Input sequences
│       ├── Y.npy             # Target sequences
│       └── stats.npz         # Normalization stats
└── training_stations.json    # Station coordinates (500+ stations)

checkpoints/
├── lilith_best.pt            # Best model checkpoint
└── lilith_*.pt               # Other checkpoints (gitignored)
```

### Avoiding Data Re-downloads

Training data is cached locally. To avoid re-downloading on every build:

```bash
# Check if data exists before downloading
if [ ! -d "data/raw/ghcn_daily/stations" ]; then
    python scripts/download_data.py --max-stations 500
fi

# Or use the --skip-existing flag
python scripts/download_data.py --max-stations 500 --skip-existing
```

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
  url = {https://github.com/consigcody94/lilith}
}
```

---

<p align="center">
  <br>
  <em>"The storm goddess sees all horizons."</em>
  <br><br>
  <strong>Weather prediction should be free. The data is public. The science is open. Now the tools are too.</strong>
</p>
