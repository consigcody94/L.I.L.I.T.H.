import pytest
import torch

from models.lilith import ForecastDecoder, LILITH, LILITHConfig


@pytest.mark.parametrize("stations", [None, 3])
@pytest.mark.parametrize("horizon", [14, 90])
def test_decoder_preserves_axes_and_stays_bounded(stations, horizon):
    torch.manual_seed(42)
    decoder = ForecastDecoder(32, 3, forecast_length=90).eval()
    shape = (2, 32) if stations is None else (2, stations, 32)
    with torch.no_grad():
        output = decoder(torch.randn(shape), n_steps=horizon)
    assert output.shape == ((2, horizon, 3) if stations is None else (2, stations, horizon, 3))
    assert torch.isfinite(output).all()
    assert output.abs().max() < 10


def test_decoder_rejects_untrained_horizon():
    decoder = ForecastDecoder(32, 3, forecast_length=14)
    with pytest.raises(ValueError):
        decoder(torch.randn(2, 32), n_steps=15)


def test_shared_graph_batch_matches_separate_graphs():
    torch.manual_seed(42)
    model = LILITH(LILITHConfig(variant="tiny", use_grid=False, use_climate_embed=False,
                               forecast_length=2)).eval()
    values, coords = torch.randn(2, 3, 4, 7), torch.randn(2, 3, 3)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    attributes = torch.ones(4, 1)
    with torch.no_grad():
        together = model(values, coords, edges, attributes)["forecast"]
        separate = torch.cat([model(values[i:i+1], coords[i:i+1], edges, attributes)["forecast"]
                              for i in range(2)])
    assert torch.allclose(together, separate, atol=1e-5, rtol=1e-5)
