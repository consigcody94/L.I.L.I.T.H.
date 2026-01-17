"""
Spherical Fourier Neural Operator (SFNO) for LILITH.

Processes atmospheric state on a spherical domain using spectral methods.
Based on NVIDIA's FourCastNet architecture.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import torch_harmonics for spherical harmonics
try:
    import torch_harmonics as th
    TORCH_HARMONICS_AVAILABLE = True
except ImportError:
    TORCH_HARMONICS_AVAILABLE = False


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution using FFT.

    Performs convolution in the Fourier domain for global receptive field
    with O(N log N) complexity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 32,
        modes2: int = 32,
    ):
        """
        Initialize spectral convolution.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Learnable Fourier coefficients
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Complex multiplication for batched inputs."""
        # (batch, in_ch, x, y) * (in_ch, out_ch, x, y) -> (batch, out_ch, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of same shape
        """
        batch_size = x.size(0)
        height, width = x.size(-2), x.size(-1)

        # Compute FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Lower modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1,
        )

        # Upper modes (for symmetry)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2,
        )

        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(height, width))

        return x


class SphericalConv(nn.Module):
    """
    Spherical convolution using spherical harmonics.

    Properly handles the geometry of the sphere, avoiding polar distortion
    that occurs with standard 2D convolutions on lat-lon grids.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nlat: int = 721,
        nlon: int = 1440,
        lmax: Optional[int] = None,
    ):
        """
        Initialize spherical convolution.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            nlat: Number of latitude points
            nlon: Number of longitude points
            lmax: Maximum spherical harmonic degree
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or nlat // 2

        if TORCH_HARMONICS_AVAILABLE:
            # Use torch_harmonics for proper spherical harmonics
            self.sht = th.RealSHT(nlat, nlon, lmax=self.lmax)
            self.isht = th.InverseRealSHT(nlat, nlon, lmax=self.lmax)

            # Learnable spectral weights
            n_coeffs = (self.lmax + 1) * (self.lmax + 2) // 2
            self.spectral_weights = nn.Parameter(
                torch.randn(in_channels, out_channels, n_coeffs) / math.sqrt(in_channels)
            )
        else:
            # Fallback to FFT-based convolution
            self.spectral_conv = SpectralConv2d(
                in_channels, out_channels,
                modes1=min(32, nlat // 2),
                modes2=min(32, nlon // 2),
            )
            self.spectral_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, nlat, nlon)

        Returns:
            Output tensor of same shape
        """
        if TORCH_HARMONICS_AVAILABLE and self.spectral_weights is not None:
            batch_size = x.size(0)

            # Transform to spectral domain
            x_spec = self.sht(x)

            # Apply learnable weights
            out_spec = torch.einsum("bixk,iok->boxk", x_spec, self.spectral_weights)

            # Transform back
            x = self.isht(out_spec)
        else:
            x = self.spectral_conv(x)

        return x


class SFNOBlock(nn.Module):
    """
    Single block of the Spherical Fourier Neural Operator.

    Combines spectral convolution with pointwise MLP and residual connection.
    """

    def __init__(
        self,
        channels: int,
        nlat: int = 64,
        nlon: int = 128,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        use_spherical: bool = True,
    ):
        """
        Initialize SFNO block.

        Args:
            channels: Number of channels
            nlat: Number of latitude points
            nlon: Number of longitude points
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout probability
            use_spherical: Use spherical harmonics (if available)
        """
        super().__init__()

        self.channels = channels
        self.use_spherical = use_spherical and TORCH_HARMONICS_AVAILABLE

        # Spectral convolution
        if self.use_spherical:
            self.spectral = SphericalConv(channels, channels, nlat, nlon)
        else:
            self.spectral = SpectralConv2d(
                channels, channels,
                modes1=min(32, nlat // 2),
                modes2=min(32, nlon // 2),
            )

        # Pointwise MLP
        hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, channels, 1),
            nn.Dropout(dropout),
        )

        # Normalization
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Spectral path
        h = self.norm1(x)
        h = self.spectral(h)
        x = x + h

        # MLP path
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h

        return x


class SphericalFourierNeuralOperator(nn.Module):
    """
    Full Spherical Fourier Neural Operator.

    A neural operator that learns atmospheric dynamics on a spherical domain
    using spectral methods for efficient global communication.

    Based on:
    - FourCastNet (NVIDIA)
    - Spherical Fourier Neural Operators (Bonev et al.)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 4,
        nlat: int = 64,
        nlon: int = 128,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        use_spherical: bool = True,
    ):
        """
        Initialize SFNO.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of SFNO blocks
            nlat: Number of latitude points in grid
            nlon: Number of longitude points in grid
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
            use_spherical: Use spherical harmonics
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlat = nlat
        self.nlon = nlon

        # Input projection
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, 1)
        self.input_norm = nn.GroupNorm(min(32, hidden_dim), hidden_dim)

        # SFNO blocks
        self.blocks = nn.ModuleList([
            SFNOBlock(
                channels=hidden_dim,
                nlat=nlat,
                nlon=nlon,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_spherical=use_spherical,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.GroupNorm(min(32, hidden_dim), hidden_dim)
        self.output_proj = nn.Conv2d(hidden_dim, output_dim, 1)

        # Gradient checkpointing
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process gridded atmospheric state.

        Args:
            x: Input tensor of shape (batch, input_dim, nlat, nlon)

        Returns:
            Output tensor of shape (batch, output_dim, nlat, nlon)
        """
        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)

        # Apply SFNO blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)

        # Output projection
        h = self.output_norm(h)
        h = self.output_proj(h)

        return h

    def forward_multiscale(
        self,
        x: torch.Tensor,
        scales: Tuple[int, ...] = (1, 2, 4),
    ) -> torch.Tensor:
        """
        Multi-scale processing for capturing different spatial patterns.

        Args:
            x: Input tensor
            scales: Downsampling factors to use

        Returns:
            Combined multi-scale output
        """
        outputs = []

        for scale in scales:
            if scale > 1:
                # Downsample
                x_scaled = F.avg_pool2d(x, scale)
                # Process
                h = self.forward(x_scaled)
                # Upsample back
                h = F.interpolate(h, size=(self.nlat, self.nlon), mode="bilinear")
            else:
                h = self.forward(x)

            outputs.append(h)

        # Combine scales (simple average, could be learned)
        return torch.stack(outputs).mean(dim=0)
