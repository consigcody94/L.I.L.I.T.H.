"""
Temporal Transformer for LILITH.

Processes temporal sequences of weather observations using
self-attention with Flash Attention optimization.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Flash Attention, fallback to standard attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position information directly into the attention mechanism
    through rotation of query and key vectors.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute rotary embeddings
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Pre-compute sin and cos for positions."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotary embeddings for sequence.

        Args:
            x: Input tensor to get seq_len from
            seq_dim: Dimension containing sequence length

        Returns:
            Tuple of (cos, sin) embeddings
        """
        seq_len = x.size(seq_dim)
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key."""
    # Expand cos/sin to match batch dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention for temporal sequences.

    Supports:
    - Flash Attention for memory efficiency
    - Rotary Position Embeddings
    - Causal masking for autoregressive prediction
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_flash: bool = True,
        use_rope: bool = True,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash and FLASH_ATTN_AVAILABLE
        self.use_rope = use_rope

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # QKV projection
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

        # Rotary embeddings
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Apply temporal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Attention mask of shape (batch, seq_len) or (batch, seq_len, seq_len)
            causal: Whether to apply causal masking

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Compute QKV
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings
        if self.rope is not None:
            cos, sin = self.rope(x)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Use Flash Attention if available
        if self.use_flash and not mask and x.is_cuda:
            # Flash attention expects (batch, seq_len, heads, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            out = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                causal=causal,
            )
            out = out.reshape(batch_size, seq_len, self.dim)
        else:
            # Standard attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply causal mask
            if causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1,
                )
                attn_weights.masked_fill_(causal_mask, float("-inf"))

            # Apply attention mask
            if mask is not None:
                if mask.dim() == 2:
                    # (batch, seq_len) -> (batch, 1, 1, seq_len)
                    mask = mask.unsqueeze(1).unsqueeze(2)
                attn_weights.masked_fill_(~mask, float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)

        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = hidden_dim or dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalTransformerBlock(nn.Module):
    """
    Single Transformer block for temporal processing.

    Consists of:
    1. Pre-norm self-attention
    2. Pre-norm feed-forward
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_flash: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash=use_flash,
            use_rope=use_rope,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, hidden_dim=ffn_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass with pre-norm residual connections."""
        x = x + self.attn(self.norm1(x), mask=mask, causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalTransformer(nn.Module):
    """
    Full Temporal Transformer encoder.

    Processes sequences of weather observations to capture temporal patterns
    and dependencies over multiple time scales.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        use_flash: bool = True,
        use_rope: bool = True,
    ):
        """
        Initialize Temporal Transformer.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Transformer hidden dimension
            output_dim: Output dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ffn_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_flash: Use Flash Attention if available
            use_rope: Use Rotary Position Embeddings
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_flash=use_flash,
                use_rope=use_rope,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Process temporal sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Attention mask of shape (batch, seq_len)
            causal: Whether to use causal attention

        Returns:
            Output tensor of shape (batch, seq_len, output_dim)
        """
        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)

        # Apply transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, mask, causal,
                    use_reentrant=False,
                )
            else:
                h = layer(h, mask=mask, causal=causal)

        # Output projection
        h = self.output_norm(h)
        h = self.output_proj(h)

        return h

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cache: Optional[list] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass with KV cache for autoregressive generation.

        Args:
            x: Input tensor (typically single token)
            cache: List of cached KV pairs from previous steps
            mask: Attention mask

        Returns:
            Output tensor and updated cache
        """
        # This would be used during inference for autoregressive rollout
        # Implementation depends on specific caching strategy
        raise NotImplementedError("KV caching not yet implemented")
