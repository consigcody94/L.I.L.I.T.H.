"""Small mask-aware station residual models; no atmospheric/hazard skill implied."""

import numpy as np
import torch
from torch import nn


def anomaly_features(arrays):
    """Each observed input is centered on its own calendar-day normal."""
    mask = arrays['X_mask']
    anomaly = np.where(mask, arrays['X'] - arrays['X_climatology'], 0)
    features = []
    for window in (1, 3, 7, 14, 30):
        count = mask[:, -window:].sum(axis=1)
        features.extend([anomaly[:, -window:].sum(axis=1)/np.maximum(count, 1), count/window])
    reverse = mask[:, ::-1]
    age = reverse.argmax(axis=1)
    last = np.take_along_axis(anomaly, (29-age)[:, None, :], axis=1)[:, 0]
    last = np.where(mask.any(axis=1), last, 0)
    age = np.where(mask.any(axis=1), age, 30)
    features.extend([last, age/30, arrays['meta'], arrays['climatology'][:, 0],
                     arrays['wet_climatology'][:, :1]])
    return np.concatenate(features, axis=1).astype(np.float32)


def constrain(values):
    result = np.asarray(values, dtype=np.float32).copy()
    high, low = result[..., 0].copy(), result[..., 1].copy()
    result[..., 0] = np.maximum(high, low)
    result[..., 1] = np.minimum(high, low)
    result[..., 2] = np.maximum(result[..., 2], 0)
    return result


def fit_ridge(train, validation):
    raw = anomaly_features(train)
    mean, scale = raw.mean(axis=0), raw.std(axis=0).clip(.1)
    def design(arrays):
        x = (anomaly_features(arrays)-mean)/scale
        return np.column_stack([x, np.ones(len(x))]).astype(np.float64)
    x, vx = design(train), design(validation)
    candidates, trials = [], []
    for alpha in (1.0, 100.0, 10000.0):
        coefficients = np.zeros((3, 14, x.shape[1]), np.float64)
        for variable in range(3):
            for lead in range(14):
                ok = train['Y_mask'][:, lead, variable]
                if not ok.any():
                    continue
                xx = x[ok]
                yy = (train['Y']-train['climatology'])[ok, lead, variable]
                penalty = np.eye(x.shape[1])*alpha
                penalty[-1, -1] = 1e-8
                coefficients[variable, lead] = np.linalg.solve(xx.T@xx+penalty, xx.T@yy)
        prediction = constrain(validation['climatology'] + np.einsum('nf,vlf->nlv', vx, coefficients))
        error = np.where(validation['Y_mask'], (prediction-validation['Y'])**2, 0)
        mse = error.sum(axis=(0, 1))/validation['Y_mask'].sum(axis=(0, 1))
        candidates.append(coefficients); trials.append({'alpha': alpha, 'validation_mse': mse.tolist()})
    selected = np.argmin(np.array([t['validation_mse'] for t in trials]), axis=0)
    coefficients = np.stack([candidates[selected[v]][v] for v in range(3)])
    return {'coefficients': coefficients, 'feature_mean': mean, 'feature_scale': scale,
            'selected_alphas': np.array([trials[i]['alpha'] for i in selected]), 'trials': trials}


def ridge_predict(arrays, fit):
    x = (anomaly_features(arrays)-fit['feature_mean'])/fit['feature_scale']
    x = np.column_stack([x, np.ones(len(x))])
    return constrain(arrays['climatology'] + np.einsum('nf,vlf->nlv', x, fit['coefficients']))


def neural_inputs(arrays, fit, scales):
    mask = arrays['X_mask']
    anomaly = np.where(mask, arrays['X']-arrays['X_climatology'], 0)/scales
    climate_scale = np.array([30, 30, 10], np.float32)
    history = np.concatenate([anomaly, mask, arrays['X_climatology']/climate_scale], axis=-1)
    stats = (anomaly_features(arrays)-fit['feature_mean'])/fit['feature_scale']
    base = ridge_predict(arrays, fit)
    future = np.concatenate([arrays['climatology']/climate_scale,
                             (base-arrays['climatology'])/scales,
                             arrays['wet_climatology'][..., None]], axis=-1)
    return tuple(np.asarray(x, np.float32) for x in (history, stats, future, base))


class ResidualStationModel(nn.Module):
    def __init__(self, kind, feature_count, scales):
        super().__init__()
        self.kind = kind
        self.register_buffer('scales', torch.as_tensor(scales, dtype=torch.float32))
        if kind == 'mlp':
            self.network = nn.Sequential(nn.Linear(feature_count+14*7, 192), nn.GELU(),
                                         nn.Dropout(.1), nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 14*4))
            nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
        elif kind == 'transformer':
            self.input_projection = nn.Linear(9, 64)
            self.position = nn.Parameter(torch.randn(1, 30, 64)*.01)
            layer = nn.TransformerEncoderLayer(64, 4, 128, .1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.query = nn.Linear(7, 64)
            self.lead = nn.Parameter(torch.randn(1, 14, 64)*.01)
            self.attention = nn.MultiheadAttention(64, 4, dropout=.1, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 4))
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        else:
            raise ValueError(kind)

    def forward(self, history, stats, future, base):
        if self.kind == 'mlp':
            raw = self.network(torch.cat([stats, future.flatten(1)], dim=1)).reshape(-1, 14, 4)
        else:
            missing = ~history[:, :, 3:6].bool().any(dim=-1)
            encoded = self.encoder(self.input_projection(history)+self.position, src_key_padding_mask=missing)
            query = self.query(future)+self.lead
            attended, _ = self.attention(query, encoded, encoded, key_padding_mask=missing, need_weights=False)
            raw = self.head(query+attended)
        point = base + raw[..., :3]*self.scales
        point = torch.stack([torch.maximum(point[..., 0], point[..., 1]),
                             torch.minimum(point[..., 0], point[..., 1]), point[..., 2].clamp_min(0)], dim=-1)
        probability = future[..., 6].clamp(.001, .999)
        logits = torch.logit(probability) + raw[..., 3]
        return point, logits


def masked_objective(point, logits, truth, mask, scales):
    """Missing labels contribute neither losses nor denominators."""
    squared = torch.square((point-truth)/scales)
    point_loss = torch.where(mask, squared, 0).sum()/mask.sum().clamp_min(1)
    wet_loss = nn.functional.binary_cross_entropy_with_logits(logits, (truth[..., 2] >= 1).float(), reduction='none')
    wet_mask = mask[..., 2]
    return point_loss + .2*torch.where(wet_mask, wet_loss, 0).sum()/wet_mask.sum().clamp_min(1)
