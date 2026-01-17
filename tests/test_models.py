"""
Tests for LILITH model components.
"""

import pytest
import torch

from models.lilith import LILITH, LILITHConfig
from models.components.station_embed import StationEmbedding, PositionalEncoding3D
from models.components.gat_encoder import GATEncoder, GATv2Layer
from models.components.temporal_transformer import TemporalTransformer, TemporalAttention
from models.components.sfno import SphericalFourierNeuralOperator, SpectralConv2d
from models.components.climate_embed import ClimateEmbedding
from models.components.ensemble_head import EnsembleHead, GaussianHead


class TestStationEmbedding:
    """Tests for station embedding module."""

    def test_positional_encoding_3d(self):
        """Test 3D positional encoding."""
        d_model = 64
        batch_size = 4
        n_stations = 10

        enc = PositionalEncoding3D(d_model)

        lat = torch.randn(batch_size, n_stations) * 45
        lon = torch.randn(batch_size, n_stations) * 90
        elev = torch.randn(batch_size, n_stations) * 1000

        output = enc(lat, lon, elev)

        assert output.shape == (batch_size, n_stations, d_model)
        assert not torch.isnan(output).any()

    def test_station_embedding_forward(self):
        """Test station embedding forward pass."""
        input_dim = 7
        hidden_dim = 128
        output_dim = 128
        batch_size = 2
        n_stations = 20
        seq_len = 30

        embed = StationEmbedding(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

        features = torch.randn(batch_size, n_stations, seq_len, input_dim)
        coords = torch.randn(batch_size, n_stations, 3)

        output = embed(features, coords)

        assert output.shape == (batch_size, n_stations, seq_len, output_dim)
        assert not torch.isnan(output).any()

    def test_station_embedding_single_timestep(self):
        """Test station embedding with single timestep."""
        input_dim = 7
        output_dim = 128
        batch_size = 2
        n_stations = 20

        embed = StationEmbedding(input_dim=input_dim, output_dim=output_dim)

        features = torch.randn(batch_size, n_stations, input_dim)
        coords = torch.randn(batch_size, n_stations, 3)

        output = embed(features, coords)

        assert output.shape == (batch_size, n_stations, output_dim)


class TestGATEncoder:
    """Tests for Graph Attention Network encoder."""

    def test_gatv2_layer(self):
        """Test single GATv2 layer."""
        in_dim = 64
        out_dim = 64
        num_nodes = 50
        num_edges = 200

        layer = GATv2Layer(in_dim=in_dim, out_dim=out_dim, num_heads=4)

        x = torch.randn(num_nodes, in_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        output = layer(x, edge_index)

        assert output.shape == (num_nodes, out_dim)
        assert not torch.isnan(output).any()

    def test_gat_encoder_forward(self):
        """Test full GAT encoder."""
        input_dim = 128
        hidden_dim = 128
        output_dim = 128
        num_nodes = 100
        num_edges = 500

        encoder = GATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
            num_heads=4,
        )

        x = torch.randn(num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 1)

        output = encoder(x, edge_index, edge_attr)

        assert output.shape == (num_nodes, output_dim)
        assert not torch.isnan(output).any()


class TestTemporalTransformer:
    """Tests for temporal transformer."""

    def test_temporal_attention(self):
        """Test temporal attention module."""
        dim = 128
        batch_size = 4
        seq_len = 30

        attn = TemporalAttention(dim=dim, num_heads=4, use_flash=False)

        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x)

        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()

    def test_temporal_transformer_forward(self):
        """Test full temporal transformer."""
        input_dim = 64
        hidden_dim = 128
        output_dim = 128
        batch_size = 2
        seq_len = 30

        transformer = TemporalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
            num_heads=4,
            use_flash=False,
        )

        x = torch.randn(batch_size, seq_len, input_dim)
        output = transformer(x)

        assert output.shape == (batch_size, seq_len, output_dim)
        assert not torch.isnan(output).any()

    def test_causal_attention(self):
        """Test causal masking in attention."""
        dim = 64
        batch_size = 2
        seq_len = 10

        attn = TemporalAttention(dim=dim, num_heads=2, use_flash=False)

        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x, causal=True)

        assert output.shape == (batch_size, seq_len, dim)


class TestSFNO:
    """Tests for Spherical Fourier Neural Operator."""

    def test_spectral_conv2d(self):
        """Test 2D spectral convolution."""
        in_ch = 32
        out_ch = 32
        batch_size = 2
        height = 32
        width = 64

        conv = SpectralConv2d(in_ch, out_ch, modes1=8, modes2=16)

        x = torch.randn(batch_size, in_ch, height, width)
        output = conv(x)

        assert output.shape == (batch_size, out_ch, height, width)
        assert not torch.isnan(output).any()

    def test_sfno_forward(self):
        """Test full SFNO."""
        input_dim = 64
        hidden_dim = 64
        output_dim = 64
        batch_size = 2
        nlat = 32
        nlon = 64

        sfno = SphericalFourierNeuralOperator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
            nlat=nlat,
            nlon=nlon,
            use_spherical=False,  # Use FFT fallback
        )

        x = torch.randn(batch_size, input_dim, nlat, nlon)
        output = sfno(x)

        assert output.shape == (batch_size, output_dim, nlat, nlon)
        assert not torch.isnan(output).any()


class TestClimateEmbedding:
    """Tests for climate embedding module."""

    def test_climate_embedding_forward(self):
        """Test climate embedding."""
        d_model = 128
        batch_size = 4
        seq_len = 30

        embed = ClimateEmbedding(d_model=d_model)

        day_of_year = torch.randint(1, 366, (batch_size, seq_len)).float()
        lat = torch.randn(batch_size) * 45
        lon = torch.randn(batch_size) * 90

        output = embed(day_of_year, lat=lat, lon=lon)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()


class TestEnsembleHead:
    """Tests for ensemble prediction head."""

    def test_gaussian_head(self):
        """Test Gaussian prediction head."""
        input_dim = 128
        output_dim = 3
        batch_size = 4
        n_stations = 20

        head = GaussianHead(input_dim, output_dim)

        x = torch.randn(batch_size, n_stations, input_dim)
        mean, std = head(x)

        assert mean.shape == (batch_size, n_stations, output_dim)
        assert std.shape == (batch_size, n_stations, output_dim)
        assert (std > 0).all()

    def test_ensemble_head_sampling(self):
        """Test ensemble sampling."""
        input_dim = 128
        output_dim = 3
        batch_size = 2
        n_samples = 10

        head = EnsembleHead(input_dim, output_dim, method="gaussian")

        x = torch.randn(batch_size, input_dim)
        samples = head.sample(x, n_samples)

        assert samples.shape == (n_samples, batch_size, output_dim)

    def test_ensemble_uncertainty(self):
        """Test uncertainty estimation."""
        input_dim = 128
        output_dim = 3
        batch_size = 4

        head = EnsembleHead(input_dim, output_dim, method="gaussian")

        x = torch.randn(batch_size, input_dim)
        mean, lower, upper = head.predict_with_uncertainty(x)

        assert mean.shape == (batch_size, output_dim)
        assert (lower <= mean).all()
        assert (upper >= mean).all()


class TestLILITH:
    """Tests for the full LILITH model."""

    @pytest.fixture
    def tiny_config(self):
        """Create tiny model config for testing."""
        return LILITHConfig(
            variant="tiny",
            hidden_dim=64,
            num_heads=2,
            gat_layers=1,
            temporal_layers=1,
            sfno_layers=1,
            use_grid=False,  # Disable grid for faster testing
            forecast_length=14,
        )

    def test_model_creation(self, tiny_config):
        """Test model instantiation."""
        model = LILITH(tiny_config)
        assert model is not None

        n_params = model.get_num_params()
        assert n_params > 0

    def test_forward_pass(self, tiny_config):
        """Test model forward pass."""
        model = LILITH(tiny_config)
        model.eval()

        batch_size = 2
        n_stations = 10
        seq_len = 30
        n_features = tiny_config.input_features

        # Create dummy inputs
        node_features = torch.randn(batch_size, n_stations, seq_len, n_features)
        node_coords = torch.randn(batch_size, n_stations, 3)
        node_coords[:, :, 0] = node_coords[:, :, 0] * 45  # lat
        node_coords[:, :, 1] = node_coords[:, :, 1] * 90  # lon
        node_coords[:, :, 2] = node_coords[:, :, 2] * 1000  # elev

        # Create simple graph
        edges_src = []
        edges_dst = []
        for i in range(n_stations):
            for j in range(i + 1, min(i + 3, n_stations)):
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.randn(len(edges_src), 1)

        with torch.no_grad():
            outputs = model(
                node_features=node_features,
                node_coords=node_coords,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

        assert "forecast" in outputs
        assert outputs["forecast"].shape == (
            batch_size,
            n_stations,
            tiny_config.forecast_length,
            tiny_config.output_features,
        )

    def test_gradient_checkpointing(self, tiny_config):
        """Test gradient checkpointing enables without error."""
        tiny_config.gradient_checkpointing = True
        model = LILITH(tiny_config)

        # Should not raise
        model.enable_gradient_checkpointing()

    def test_save_load(self, tiny_config, tmp_path):
        """Test model save and load."""
        model = LILITH(tiny_config)

        # Save
        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))

        # Load
        loaded_model = LILITH.from_pretrained(str(save_path))

        # Compare parameters
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
