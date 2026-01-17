"""
Training Loop for LILITH.

Supports:
- Mixed precision (FP16/BF16)
- Gradient checkpointing
- DeepSpeed integration
- Curriculum learning
- Checkpointing and resumption
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from loguru import logger

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Batch size
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = field(init=False)

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # Memory optimization
    gradient_checkpointing: bool = True

    # Distributed training
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    use_fsdp: bool = False

    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_stages: List[int] = field(default_factory=lambda: [7, 14, 30, 60, 90])
    curriculum_switch_steps: List[int] = field(default_factory=lambda: [10000, 30000, 60000, 80000])

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500
    keep_last_n_checkpoints: int = 5

    # Logging
    log_every_n_steps: int = 10
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Data
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps


class LRScheduler:
    """
    Learning rate scheduler with warmup and cosine decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._step = 0

    def step(self):
        """Update learning rate."""
        self._step += 1

        if self._step < self.warmup_steps:
            # Linear warmup
            factor = self._step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self._step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(1.0, progress)
            factor = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class Trainer:
    """
    Training loop for LILITH.

    Handles:
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing
    - Curriculum learning
    - Distributed training (DeepSpeed/FSDP)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            loss_fn: Loss function
        """
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            logger.info("Enabled gradient checkpointing")

        # Set up model
        self.model = model.to(self.device)

        # Set up loss function
        if loss_fn is None:
            from models.losses import LILITHLoss
            self.loss_fn = LILITHLoss()
        else:
            self.loss_fn = loss_fn

        # Set up optimizer
        self.optimizer = self._create_optimizer()

        # Set up learning rate scheduler
        self.lr_scheduler = LRScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )

        # Set up mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        self.amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.curriculum_stage = 0

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.metrics_history: List[Dict[str, Any]] = []

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Try to use 8-bit Adam if available
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
            )
            logger.info("Using 8-bit AdamW optimizer")
        except ImportError:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
            )
            logger.info("Using standard AdamW optimizer")

        return optimizer

    def get_current_forecast_length(self) -> int:
        """Get forecast length for current curriculum stage."""
        if not self.config.curriculum_enabled:
            return self.config.curriculum_stages[-1]

        for i, switch_step in enumerate(self.config.curriculum_switch_steps):
            if self.global_step < switch_step:
                return self.config.curriculum_stages[i]

        return self.config.curriculum_stages[-1]

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of data

        Returns:
            Dict of loss values
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Get current forecast length from curriculum
        forecast_len = self.get_current_forecast_length()

        # Forward pass with mixed precision
        with autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
            # Forward pass
            outputs = self.model(
                node_features=batch["node_features"],
                node_coords=batch["node_coords"],
                edge_index=batch["edge_index"],
                edge_attr=batch.get("edge_attr"),
                mask=batch.get("mask"),
            )

            # Compute loss
            pred = outputs["forecast"]
            target = batch["target_features"]

            # Truncate to current curriculum length
            if pred.size(-2) > forecast_len:
                pred = pred[..., :forecast_len, :]
                target = target[..., :forecast_len, :]

            losses = self.loss_fn(
                pred=pred,
                target=target,
                mask=batch.get("target_mask"),
            )
            loss = losses["total"] / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation loop."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_losses = {}
        n_batches = 0

        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    node_features=batch["node_features"],
                    node_coords=batch["node_coords"],
                    edge_index=batch["edge_index"],
                    edge_attr=batch.get("edge_attr"),
                    mask=batch.get("mask"),
                )

                losses = self.loss_fn(
                    pred=outputs["forecast"],
                    target=batch["target_features"],
                    mask=batch.get("target_mask"),
                )

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                total_losses[k] = total_losses.get(k, 0.0) + val

            n_batches += 1

        # Average losses
        avg_losses = {f"val_{k}": v / n_batches for k, v in total_losses.items()}

        return avg_losses

    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "curriculum_stage": self.curriculum_stage,
            "config": self.config.__dict__,
        }

        # Save step checkpoint
        path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the latest N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        while len(checkpoints) > self.config.keep_last_n_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            logger.debug(f"Removed old checkpoint: {oldest}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.curriculum_stage = checkpoint.get("curriculum_stage", 0)

        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")

    def train(self, resume_from: Optional[str] = None):
        """
        Run training loop.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info(f"Starting training from step {self.global_step}")
        logger.info(f"Effective batch size: {self.config.effective_batch_size}")

        # Training loop
        accumulated_losses = {}
        accumulated_steps = 0

        data_iter = iter(self.train_dataloader)

        while self.global_step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Training step
            step_losses = self.train_step(batch)
            accumulated_steps += 1

            # Accumulate losses
            for k, v in step_losses.items():
                accumulated_losses[k] = accumulated_losses.get(k, 0.0) + v

            # Optimizer step
            if accumulated_steps >= self.config.gradient_accumulation_steps:
                grad_norm = self.optimizer_step()
                self.global_step += 1
                accumulated_steps = 0

                # Log metrics
                if self.global_step % self.config.log_every_n_steps == 0:
                    avg_losses = {k: v / self.config.gradient_accumulation_steps for k, v in accumulated_losses.items()}
                    metrics = {
                        **avg_losses,
                        "grad_norm": grad_norm,
                        "lr": self.lr_scheduler.get_lr(),
                        "forecast_length": self.get_current_forecast_length(),
                        "step": self.global_step,
                        "epoch": self.epoch,
                    }
                    self._log_metrics(metrics)
                    accumulated_losses = {}

                # Validation
                if self.global_step % self.config.eval_every_n_steps == 0:
                    val_metrics = self.validate()
                    self._log_metrics(val_metrics, prefix="val")

                    # Check for best model
                    val_loss = val_metrics.get("val_total", float("inf"))
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss

                    # Save checkpoint
                    self.save_checkpoint(is_best=is_best)

                # Regular checkpoint saving
                elif self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint()

        logger.success(f"Training complete! Final step: {self.global_step}")

    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = "train"):
        """Log metrics to console and optionally to W&B."""
        # Format for console
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        logger.info(f"[{prefix}] Step {self.global_step}: {metrics_str}")

        # Store in history
        self.metrics_history.append({**metrics, "prefix": prefix, "timestamp": time.time()})

        # Log to W&B if configured
        if self.config.wandb_project:
            try:
                import wandb
                wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=self.global_step)
            except ImportError:
                pass


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train LILITH model")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=100000, help="Max training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model-variant", type=str, default="base", choices=["tiny", "base", "large", "xl"])
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    args = parser.parse_args()

    # Create model
    from models.lilith import LILITH, LILITHConfig

    model_config = LILITHConfig(variant=args.model_variant)
    model = LILITH(model_config)

    logger.info(f"Model parameters: {model.get_num_params():,}")

    # Create data loaders
    from data.loaders import ForecastDataset, collate_variable_graphs

    train_dataset = ForecastDataset(
        data_dir=args.data_dir,
        sequence_length=30,
        forecast_length=90,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_variable_graphs,
        pin_memory=True,
    )

    # Create training config
    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project="lilith" if args.wandb else None,
    )

    # Create trainer and train
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
    )

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
