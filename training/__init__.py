"""Training utilities for LILITH."""

def __getattr__(name):
    # Data preparation and portable experiments do not require the full trainer.
    if name in {"Trainer", "TrainingConfig"}:
        from training.trainer import Trainer, TrainingConfig
        return {"Trainer": Trainer, "TrainingConfig": TrainingConfig}[name]
    raise AttributeError(name)

__all__ = ["Trainer", "TrainingConfig"]
