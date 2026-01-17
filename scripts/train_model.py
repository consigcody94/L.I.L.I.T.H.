#!/usr/bin/env python3
"""
LILITH Model Training Script

Trains the LILITH weather prediction model.

Usage:
    python scripts/train_model.py --config models/configs/base.yaml --data data/storage/parquet
"""

import argparse
from pathlib import Path
from loguru import logger
import yaml

import torch
from torch.utils.data import DataLoader

from models.lilith import LILITH, LILITHConfig
from training.trainer import Trainer, TrainingConfig
from data.loaders import ForecastDataset, collate_variable_graphs


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train LILITH weather prediction model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/configs/base.yaml",
        help="Path to model configuration YAML",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps from config",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lilith",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="W&B run name",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Create model config
    model_cfg = config.get("model", {})
    model_config = LILITHConfig(
        variant=model_cfg.get("variant", "base"),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_heads=model_cfg.get("num_heads", 8),
        gat_layers=model_cfg.get("gat_layers", 3),
        temporal_layers=model_cfg.get("temporal_layers", 6),
        sfno_layers=model_cfg.get("sfno_layers", 4),
        use_grid=model_cfg.get("use_grid", True),
        nlat=model_cfg.get("nlat", 64),
        nlon=model_cfg.get("nlon", 128),
        forecast_length=model_cfg.get("forecast_length", 90),
        dropout=model_cfg.get("dropout", 0.1),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", True),
    )

    # Create model
    logger.info(f"Creating LILITH model (variant: {model_config.variant})")
    model = LILITH(model_config)
    logger.info(f"Model parameters: {model.get_num_params():,}")

    # Create training config
    train_cfg = config.get("training", {})
    training_config = TrainingConfig(
        learning_rate=args.lr or train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        warmup_steps=train_cfg.get("warmup_steps", 1000),
        max_steps=args.max_steps or train_cfg.get("max_steps", 100000),
        batch_size=args.batch_size or train_cfg.get("batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        use_amp=train_cfg.get("use_amp", True),
        amp_dtype=train_cfg.get("amp_dtype", "float16"),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        curriculum_enabled=train_cfg.get("curriculum_enabled", True),
        curriculum_stages=train_cfg.get("curriculum_stages", [7, 14, 30, 60, 90]),
        curriculum_switch_steps=train_cfg.get("curriculum_switch_steps", [10000, 30000, 60000, 80000]),
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_run_name=args.wandb_run,
    )

    # Create dataset
    logger.info(f"Loading data from {args.data}")
    train_dataset = ForecastDataset(
        data_dir=args.data,
        sequence_length=model_config.sequence_length,
        forecast_length=model_config.forecast_length,
        target_variables=["TMAX", "TMIN", "PRCP"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=collate_variable_graphs,
        pin_memory=training_config.pin_memory,
        drop_last=True,
    )

    logger.info(f"Training samples: {len(train_dataset)}")

    # Initialize W&B if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run,
                config={
                    "model": model_config.__dict__,
                    "training": training_config.__dict__,
                },
            )
            logger.info("Initialized Weights & Biases logging")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")

    # Create trainer and train
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
    )

    logger.info("Starting training...")
    trainer.train(resume_from=args.resume)

    logger.success("Training complete!")


if __name__ == "__main__":
    main()
