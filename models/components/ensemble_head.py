"""
Ensemble Head for LILITH.

Generates ensemble forecasts for uncertainty quantification using
diffusion-based sampling or dropout-based approaches.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianHead(nn.Module):
    """
    Predicts mean and variance for Gaussian output distribution.

    Simple but effective for uncertainty estimation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        min_std: float = 0.01,
        max_std: float = 10.0,
    ):
        """
        Initialize Gaussian prediction head.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of predicted variables)
            hidden_dim: Hidden layer dimension
            min_std: Minimum standard deviation
            max_std: Maximum standard deviation
        """
        super().__init__()

        self.output_dim = output_dim
        self.min_std = min_std
        self.max_std = max_std

        hidden_dim = hidden_dim or input_dim

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        # Mean prediction
        self.mean_head = nn.Linear(hidden_dim, output_dim)

        # Log variance prediction (for numerical stability)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and standard deviation.

        Args:
            x: Input features of shape (batch, ..., input_dim)

        Returns:
            Tuple of (mean, std) each of shape (batch, ..., output_dim)
        """
        h = self.shared(x)

        mean = self.mean_head(h)
        logvar = self.logvar_head(h)

        # Convert to std with bounds
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, self.min_std, self.max_std)

        return mean, std

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate samples from the predicted distribution.

        Args:
            x: Input features
            n_samples: Number of samples to generate

        Returns:
            Samples of shape (n_samples, batch, ..., output_dim)
        """
        mean, std = self.forward(x)

        # Expand for multiple samples
        mean = mean.unsqueeze(0).expand(n_samples, *mean.shape)
        std = std.unsqueeze(0).expand(n_samples, *std.shape)

        # Sample
        eps = torch.randn_like(mean)
        samples = mean + std * eps

        return samples


class QuantileHead(nn.Module):
    """
    Predicts multiple quantiles for non-Gaussian distributions.

    Useful for skewed variables like precipitation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        quantiles: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize quantile prediction head.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of predicted variables)
            quantiles: Quantile levels to predict
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.output_dim = output_dim
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)

        hidden_dim = hidden_dim or input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim * self.n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict quantiles.

        Args:
            x: Input features of shape (batch, ..., input_dim)

        Returns:
            Quantiles of shape (batch, ..., output_dim, n_quantiles)
        """
        shape = x.shape[:-1]
        out = self.net(x)

        # Reshape to separate quantiles
        out = out.view(*shape, self.output_dim, self.n_quantiles)

        # Ensure quantiles are monotonically increasing
        # Using softmax to get positive increments
        increments = F.softmax(out, dim=-1)
        out = torch.cumsum(increments, dim=-1)

        return out


class MCDropoutHead(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.

    Uses dropout at inference time to generate ensemble samples.
    Simple and computationally efficient.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        n_layers: int = 2,
    ):
        """
        Initialize MC Dropout head.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            n_layers: Number of hidden layers
        """
        super().__init__()

        self.output_dim = output_dim
        self.dropout = dropout

        hidden_dim = hidden_dim or input_dim

        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.net(x)

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> torch.Tensor:
        """
        Generate samples using MC Dropout.

        Args:
            x: Input features
            n_samples: Number of samples to generate

        Returns:
            Samples of shape (n_samples, batch, ..., output_dim)
        """
        # Ensure dropout is active
        self.train()

        samples = []
        for _ in range(n_samples):
            samples.append(self.forward(x))

        return torch.stack(samples, dim=0)


class DiffusionEnsembleHead(nn.Module):
    """
    Diffusion-based ensemble generation for high-quality uncertainty.

    Uses a lightweight denoising diffusion model to generate diverse
    ensemble members conditioned on the deterministic forecast.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """
        Initialize diffusion ensemble head.

        Args:
            input_dim: Input (conditioning) feature dimension
            output_dim: Output dimension
            hidden_dim: Hidden dimension for denoising network
            n_steps: Number of diffusion steps
            beta_start: Starting noise schedule
            beta_end: Ending noise schedule
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_steps = n_steps

        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Denoising network (simple MLP)
        self.denoise_net = nn.Sequential(
            nn.Linear(output_dim + input_dim + hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

        # Mean prediction (deterministic baseline)
        self.mean_head = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic mean prediction."""
        return self.mean_head(x)

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single denoising step.

        Args:
            x_t: Noisy sample at time t
            t: Time step (normalized to [0, 1])
            condition: Conditioning information

        Returns:
            Predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))

        # Concatenate inputs
        h = torch.cat([x_t, condition, t_emb], dim=-1)

        # Predict noise
        return self.denoise_net(h)

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> torch.Tensor:
        """
        Generate ensemble samples via reverse diffusion.

        Args:
            x: Conditioning features of shape (batch, ..., input_dim)
            n_samples: Number of ensemble members

        Returns:
            Samples of shape (n_samples, batch, ..., output_dim)
        """
        shape = x.shape[:-1]
        device = x.device

        samples = []

        for _ in range(n_samples):
            # Start from noise
            x_t = torch.randn(*shape, self.output_dim, device=device)

            # Reverse diffusion
            for i in reversed(range(self.n_steps)):
                t = torch.full(shape, i / self.n_steps, device=device)

                # Predict noise
                noise_pred = self.denoise_step(x_t, t, x)

                # Denoise step
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]

                if i > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0

                x_t = (
                    1 / torch.sqrt(alpha) *
                    (x_t - beta / self.sqrt_one_minus_alphas_cumprod[i] * noise_pred)
                    + torch.sqrt(beta) * noise
                )

            # Add deterministic mean
            mean = self.mean_head(x)
            samples.append(x_t + mean)

        return torch.stack(samples, dim=0)


class EnsembleHead(nn.Module):
    """
    Unified ensemble head that combines multiple uncertainty methods.

    Supports:
    - Gaussian parametric uncertainty
    - Quantile regression
    - MC Dropout
    - Diffusion ensemble (optional)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        method: str = "gaussian",  # "gaussian", "quantile", "mc_dropout", "diffusion"
        n_quantiles: int = 5,
        dropout: float = 0.1,
        diffusion_steps: int = 50,
    ):
        """
        Initialize ensemble head.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dim: Hidden dimension
            method: Uncertainty method to use
            n_quantiles: Number of quantiles (for quantile method)
            dropout: Dropout rate (for MC dropout)
            diffusion_steps: Diffusion steps (for diffusion method)
        """
        super().__init__()

        self.method = method
        self.output_dim = output_dim

        if method == "gaussian":
            self.head = GaussianHead(input_dim, output_dim, hidden_dim)
        elif method == "quantile":
            quantiles = tuple([i / (n_quantiles + 1) for i in range(1, n_quantiles + 1)])
            self.head = QuantileHead(input_dim, output_dim, quantiles, hidden_dim)
        elif method == "mc_dropout":
            self.head = MCDropoutHead(input_dim, output_dim, hidden_dim, dropout)
        elif method == "diffusion":
            self.head = DiffusionEnsembleHead(
                input_dim, output_dim, hidden_dim, diffusion_steps
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get deterministic prediction.

        Args:
            x: Input features

        Returns:
            Prediction (mean for Gaussian, median for quantile, etc.)
        """
        if self.method == "gaussian":
            mean, _ = self.head(x)
            return mean
        elif self.method == "quantile":
            quantiles = self.head(x)
            # Return median (middle quantile)
            return quantiles[..., quantiles.size(-1) // 2]
        else:
            return self.head(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get prediction with uncertainty estimates.

        Args:
            x: Input features
            n_samples: Number of samples for MC methods

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        if self.method == "gaussian":
            mean, std = self.head(x)
            lower = mean - 1.96 * std  # 95% CI
            upper = mean + 1.96 * std
            return mean, lower, upper

        elif self.method == "quantile":
            quantiles = self.head(x)
            mean = quantiles[..., quantiles.size(-1) // 2]
            lower = quantiles[..., 0]  # Lowest quantile
            upper = quantiles[..., -1]  # Highest quantile
            return mean, lower, upper

        else:
            # MC methods
            samples = self.head.sample(x, n_samples)
            mean = samples.mean(dim=0)
            lower = samples.quantile(0.025, dim=0)
            upper = samples.quantile(0.975, dim=0)
            return mean, lower, upper

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> torch.Tensor:
        """
        Generate ensemble samples.

        Args:
            x: Input features
            n_samples: Number of samples

        Returns:
            Ensemble samples
        """
        if hasattr(self.head, "sample"):
            return self.head.sample(x, n_samples)
        elif self.method == "gaussian":
            return self.head.sample(x, n_samples)
        else:
            # For quantile, sample uniformly between quantiles
            quantiles = self.head(x)
            samples = []
            for _ in range(n_samples):
                # Random interpolation between adjacent quantiles
                idx = torch.randint(0, quantiles.size(-1) - 1, (1,)).item()
                alpha = torch.rand(1, device=x.device)
                sample = (1 - alpha) * quantiles[..., idx] + alpha * quantiles[..., idx + 1]
                samples.append(sample)
            return torch.stack(samples, dim=0)
