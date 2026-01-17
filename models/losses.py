"""
Loss Functions for LILITH.

Multi-task losses optimized for weather forecasting:
- Temperature: MSE with gradient consistency
- Precipitation: BCE (occurrence) + Quantile (amount)
- Uncertainty: Negative log-likelihood
- Spectral: Spatial coherence
"""

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss with support for:
    - Variable-specific weights
    - Temporal decay (weight early timesteps more)
    - Missing value handling via mask
    """

    def __init__(
        self,
        var_weights: Optional[Dict[str, float]] = None,
        temporal_decay: float = 0.0,
    ):
        """
        Initialize weighted MSE loss.

        Args:
            var_weights: Dictionary mapping variable names to weights
            temporal_decay: Exponential decay factor for temporal weights
        """
        super().__init__()

        self.var_weights = var_weights or {}
        self.temporal_decay = temporal_decay

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        var_names: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Args:
            pred: Predictions of shape (batch, ..., n_vars)
            target: Targets of same shape
            mask: Valid value mask of same shape
            var_names: Variable names for per-variable weighting

        Returns:
            Scalar loss value
        """
        # Compute squared errors
        sq_errors = (pred - target) ** 2

        # Apply mask
        if mask is not None:
            sq_errors = sq_errors * mask.float()
            n_valid = mask.sum()
        else:
            n_valid = sq_errors.numel()

        # Apply variable weights
        if var_names and self.var_weights:
            weights = torch.tensor(
                [self.var_weights.get(name, 1.0) for name in var_names],
                device=pred.device,
            )
            sq_errors = sq_errors * weights

        # Apply temporal decay
        if self.temporal_decay > 0 and pred.dim() >= 3:
            n_timesteps = pred.size(-2)
            decay = torch.exp(-self.temporal_decay * torch.arange(n_timesteps, device=pred.device))
            decay = decay / decay.sum() * n_timesteps  # Normalize
            sq_errors = sq_errors * decay.view(1, -1, 1)

        # Compute mean
        loss = sq_errors.sum() / (n_valid + 1e-8)

        return loss


class GradientConsistencyLoss(nn.Module):
    """
    Penalizes unrealistic temporal gradients in forecasts.

    Encourages smooth, physically plausible transitions.
    """

    def __init__(self, max_daily_change: float = 20.0):
        """
        Initialize gradient consistency loss.

        Args:
            max_daily_change: Maximum allowed daily change (e.g., 20°C)
        """
        super().__init__()
        self.max_daily_change = max_daily_change

    def forward(
        self,
        pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gradient consistency loss.

        Args:
            pred: Predictions of shape (batch, ..., n_timesteps, n_vars)
            mask: Valid value mask

        Returns:
            Scalar loss
        """
        # Compute temporal gradients
        gradients = pred[..., 1:, :] - pred[..., :-1, :]

        # Penalize gradients exceeding threshold
        excess = F.relu(gradients.abs() - self.max_daily_change)

        if mask is not None:
            # Mask needs to be aligned for gradients
            grad_mask = mask[..., 1:, :] & mask[..., :-1, :]
            excess = excess * grad_mask.float()
            n_valid = grad_mask.sum()
        else:
            n_valid = excess.numel()

        return excess.sum() / (n_valid + 1e-8)


class PrecipitationLoss(nn.Module):
    """
    Combined loss for precipitation prediction:
    1. BCE for occurrence (rain/no-rain)
    2. Quantile loss for amount (when raining)

    Handles the mixed discrete-continuous nature of precipitation.
    """

    def __init__(
        self,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        occurrence_weight: float = 1.0,
        amount_weight: float = 1.0,
        threshold: float = 0.1,  # mm, threshold for "rain"
    ):
        """
        Initialize precipitation loss.

        Args:
            quantiles: Quantile levels for amount prediction
            occurrence_weight: Weight for occurrence loss
            amount_weight: Weight for amount loss
            threshold: Precipitation threshold for occurrence
        """
        super().__init__()

        self.quantiles = quantiles
        self.occurrence_weight = occurrence_weight
        self.amount_weight = amount_weight
        self.threshold = threshold

    def forward(
        self,
        pred_occurrence: torch.Tensor,
        pred_amount: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute precipitation loss.

        Args:
            pred_occurrence: Predicted probability of rain (batch, ...)
            pred_amount: Predicted amount quantiles (batch, ..., n_quantiles)
            target: Target precipitation (batch, ...)
            mask: Valid value mask

        Returns:
            Tuple of (occurrence_loss, amount_loss)
        """
        # Occurrence (binary classification)
        target_occurrence = (target > self.threshold).float()
        occurrence_loss = F.binary_cross_entropy(
            pred_occurrence,
            target_occurrence,
            reduction="none",
        )

        if mask is not None:
            occurrence_loss = occurrence_loss * mask.float()
            n_occ_valid = mask.sum()
        else:
            n_occ_valid = occurrence_loss.numel()

        occurrence_loss = occurrence_loss.sum() / (n_occ_valid + 1e-8)

        # Amount (quantile regression, only for rainy samples)
        rain_mask = target > self.threshold
        if mask is not None:
            rain_mask = rain_mask & mask

        if rain_mask.sum() > 0:
            # Log-transform precipitation for better distribution
            target_log = torch.log1p(target)

            # Quantile loss
            amount_loss = 0.0
            for i, q in enumerate(self.quantiles):
                pred_q = pred_amount[..., i]
                errors = target_log - pred_q
                quantile_loss = torch.where(
                    errors >= 0,
                    q * errors,
                    (1 - q) * (-errors),
                )
                quantile_loss = quantile_loss * rain_mask.float()
                amount_loss = amount_loss + quantile_loss.sum()

            amount_loss = amount_loss / (rain_mask.sum() * len(self.quantiles) + 1e-8)
        else:
            amount_loss = torch.tensor(0.0, device=pred_amount.device)

        return (
            self.occurrence_weight * occurrence_loss,
            self.amount_weight * amount_loss,
        )


class GaussianNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for Gaussian predictions.

    Used for probabilistic forecasts with mean and variance.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL.

        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            target: Target values
            mask: Valid value mask

        Returns:
            Scalar loss
        """
        var = std ** 2 + self.eps

        # NLL = 0.5 * (log(var) + (target - mean)^2 / var)
        nll = 0.5 * (torch.log(var) + (target - mean) ** 2 / var)

        if mask is not None:
            nll = nll * mask.float()
            n_valid = mask.sum()
        else:
            n_valid = nll.numel()

        return nll.sum() / (n_valid + 1e-8)


class CRPSLoss(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS) loss.

    Proper scoring rule for probabilistic forecasts.
    Lower is better.
    """

    def __init__(self, n_samples: int = 100):
        """
        Initialize CRPS loss.

        Args:
            n_samples: Number of samples for Monte Carlo estimation
        """
        super().__init__()
        self.n_samples = n_samples

    def forward(
        self,
        samples: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute CRPS via Monte Carlo.

        Args:
            samples: Ensemble samples (n_samples, batch, ...)
            target: Target values (batch, ...)
            mask: Valid value mask

        Returns:
            Scalar CRPS value
        """
        n_samples = samples.size(0)

        # Term 1: E[|X - y|]
        term1 = (samples - target.unsqueeze(0)).abs().mean(dim=0)

        # Term 2: E[|X - X'|] / 2
        # Approximate with pairs of samples
        idx1 = torch.randperm(n_samples, device=samples.device)
        idx2 = torch.randperm(n_samples, device=samples.device)
        term2 = (samples[idx1] - samples[idx2]).abs().mean(dim=0) / 2

        crps = term1 - term2

        if mask is not None:
            crps = crps * mask.float()
            n_valid = mask.sum()
        else:
            n_valid = crps.numel()

        return crps.sum() / (n_valid + 1e-8)


class SpectralLoss(nn.Module):
    """
    Spectral loss for spatial coherence.

    Encourages physically realistic spatial patterns in forecasts.
    """

    def __init__(self, weight_high_freq: float = 0.1):
        """
        Initialize spectral loss.

        Args:
            weight_high_freq: Weight for high-frequency components
        """
        super().__init__()
        self.weight_high_freq = weight_high_freq

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute spectral loss on gridded data.

        Args:
            pred: Predictions of shape (batch, channels, height, width)
            target: Targets of same shape

        Returns:
            Scalar loss
        """
        # Compute 2D FFT
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        # Compute magnitude difference
        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()

        diff = (pred_mag - target_mag) ** 2

        # Weight high frequencies less (they're often noise)
        h, w = diff.shape[-2:]
        freq_weight = 1.0 / (1.0 + self.weight_high_freq * (
            torch.arange(h, device=pred.device).view(-1, 1) ** 2 +
            torch.arange(w, device=pred.device).view(1, -1) ** 2
        ).sqrt())

        diff = diff * freq_weight

        return diff.mean()


class LILITHLoss(nn.Module):
    """
    Combined loss function for LILITH model.

    Combines:
    - Temperature MSE
    - Precipitation occurrence + amount
    - Gradient consistency
    - Optional: Spectral loss, CRPS
    """

    def __init__(
        self,
        temp_weight: float = 1.0,
        precip_weight: float = 2.0,
        gradient_weight: float = 0.1,
        spectral_weight: float = 0.0,
        uncertainty_weight: float = 0.1,
        var_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize combined loss.

        Args:
            temp_weight: Weight for temperature loss
            precip_weight: Weight for precipitation loss
            gradient_weight: Weight for gradient consistency
            spectral_weight: Weight for spectral loss
            uncertainty_weight: Weight for uncertainty calibration
            var_weights: Per-variable weights
        """
        super().__init__()

        self.temp_weight = temp_weight
        self.precip_weight = precip_weight
        self.gradient_weight = gradient_weight
        self.spectral_weight = spectral_weight
        self.uncertainty_weight = uncertainty_weight

        # Component losses
        self.mse_loss = WeightedMSELoss(var_weights)
        self.gradient_loss = GradientConsistencyLoss()
        self.precip_loss = PrecipitationLoss()
        self.nll_loss = GaussianNLLLoss()

        if spectral_weight > 0:
            self.spectral_loss = SpectralLoss()
        else:
            self.spectral_loss = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pred_std: Optional[torch.Tensor] = None,
        var_names: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred: Predictions (batch, ..., n_vars)
            target: Targets (batch, ..., n_vars)
            mask: Valid value mask
            pred_std: Predicted uncertainty (for NLL loss)
            var_names: Variable names

        Returns:
            Dict with "total" and component losses
        """
        losses = {}

        # Default variable names
        if var_names is None:
            var_names = ["TMAX", "TMIN", "PRCP"]

        # Temperature loss (TMAX, TMIN)
        temp_vars = [i for i, name in enumerate(var_names) if "T" in name]
        if temp_vars:
            temp_pred = pred[..., temp_vars]
            temp_target = target[..., temp_vars]
            temp_mask = mask[..., temp_vars] if mask is not None else None

            losses["temp_mse"] = self.mse_loss(temp_pred, temp_target, temp_mask)

        # Precipitation loss
        precip_vars = [i for i, name in enumerate(var_names) if "PRCP" in name or "precip" in name.lower()]
        if precip_vars:
            # For now, treat prediction as mean, skip occurrence model
            precip_pred = pred[..., precip_vars[0]]
            precip_target = target[..., precip_vars[0]]
            precip_mask = mask[..., precip_vars[0]] if mask is not None else None

            # Simple MSE for precipitation (log-transformed)
            precip_pred_log = torch.log1p(F.relu(precip_pred))
            precip_target_log = torch.log1p(precip_target.clamp(min=0))

            if precip_mask is not None:
                precip_diff = (precip_pred_log - precip_target_log) ** 2 * precip_mask.float()
                losses["precip_mse"] = precip_diff.sum() / (precip_mask.sum() + 1e-8)
            else:
                losses["precip_mse"] = F.mse_loss(precip_pred_log, precip_target_log)

        # Gradient consistency
        if self.gradient_weight > 0:
            losses["gradient"] = self.gradient_loss(pred, mask)

        # Uncertainty loss (NLL)
        if pred_std is not None and self.uncertainty_weight > 0:
            losses["nll"] = self.nll_loss(pred, pred_std, target, mask)

        # Compute total loss
        total = 0.0
        if "temp_mse" in losses:
            total = total + self.temp_weight * losses["temp_mse"]
        if "precip_mse" in losses:
            total = total + self.precip_weight * losses["precip_mse"]
        if "gradient" in losses:
            total = total + self.gradient_weight * losses["gradient"]
        if "nll" in losses:
            total = total + self.uncertainty_weight * losses["nll"]

        losses["total"] = total

        return losses
