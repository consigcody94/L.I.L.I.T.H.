"""
SimpleLILITH - Simplified single-station sequence prediction model.

Canonical implementation shared between training and inference.
Uses an encoder-decoder Transformer architecture with additive
station metadata embedding and causal decoding.
"""

import numpy as np
import torch
import torch.nn as nn


class SimpleLILITH(nn.Module):
    """
    Simplified LILITH for single-station sequence prediction.

    Architecture:
        - Input projection + sinusoidal positional encoding
        - Station metadata embedding (additive)
        - Transformer encoder with GELU activation
        - Transformer decoder with causal mask
        - Output projection to forecast features
    """

    def __init__(
        self,
        input_features: int = 3,
        output_features: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        max_forecast: int = 90,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_forecast = max_forecast
        self.output_features = output_features

        # Input projection
        self.input_proj = nn.Linear(input_features, d_model)

        # Station metadata embedding
        self.meta_embed = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding
        self.register_buffer("pos_encoding", self._create_pe(500, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_features),
        )

        # Initialize weights
        self._init_weights()

    def _create_pe(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self):
        """Xavier uniform initialization for all weight matrices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        meta: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch, seq_len, features]
            meta: Station metadata [batch, 4] (lat, lon, elevation, day_of_year)
            target_len: Number of days to forecast

        Returns:
            Forecast tensor [batch, target_len, output_features]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(device)

        # Add station embedding (additive, not concatenative)
        station_emb = self.meta_embed(meta)
        x = x + station_emb.unsqueeze(1)

        # Encode
        memory = self.encoder(x)

        # Create decoder queries with positional encoding and station context
        tgt = torch.zeros(batch_size, target_len, self.d_model, device=device)
        tgt = tgt + self.pos_encoding[:, :target_len, :].to(device)
        tgt = tgt + station_emb.unsqueeze(1)

        # Create causal mask for autoregressive decoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            target_len, device=device
        )

        # Decode
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        # Project to output features
        return self.output_proj(output)

    @torch.no_grad()
    def mc_dropout_forecast(
        self,
        x: torch.Tensor,
        meta: torch.Tensor,
        target_len: int,
        n_samples: int = 30,
    ) -> torch.Tensor:
        """Cheap epistemic-uncertainty proxy via Monte Carlo Dropout.

        Keeps dropout layers active during inference and runs the model
        ``n_samples`` times. Returns ``(n_samples, batch, target_len, out_features)``.
        This is a strict upgrade over the noise-injection trick the original
        repo shipped: samples are genuinely decorrelated and reflect actual
        weight-space uncertainty.

        Caveats — read these before using the intervals operationally:
            * MC Dropout is *not* a faithful Bayesian posterior. Le Folgoc
              et al. ("Is MC Dropout Bayesian?", arXiv:2110.04286) show that
              its predictive posterior assigns 0 probability to the true
              model on closed-form benchmarks; multimodality is a design
              artefact, not real epistemic structure.
            * Empirical calibration of MC Dropout intervals is mixed —
              recent work finds only modest correlation between MC Dropout
              uncertainty and actual prediction error (e.g. r ~ 0.3–0.4 in
              segmentation; arXiv:2510.15541). Calibrate against held-out
              data before reporting confidence bands to users.
            * For a production-grade ensemble, train a *deep ensemble*
              (5-10 independently-initialized models) and average. Costs
              more compute but gives genuinely calibrated uncertainty.

        Cost: n_samples × the deterministic forward pass.
        """
        was_training = self.training
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        try:
            samples = [self.forward(x, meta, target_len) for _ in range(n_samples)]
        finally:
            self.train(was_training)

        return torch.stack(samples, dim=0)
