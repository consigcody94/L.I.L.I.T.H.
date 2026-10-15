"""Observed-target metrics, empirical interval checks and clustered comparisons."""

import numpy as np


def safe_divide(total, count):
    return np.divide(total, count, out=np.full(np.shape(total), np.nan, dtype=float), where=count > 0)


def scores(prediction, arrays):
    mask = arrays['Y_mask']
    error = np.where(mask, prediction-arrays['Y'], 0).astype(np.float64)
    count = mask.sum(axis=(0, 1))
    return {'cases': len(mask), 'stations': len(set(arrays['station_id'])), 'observed_targets': count.tolist(),
            'mae': safe_divide(abs(error).sum(axis=(0, 1)), count).tolist(),
            'rmse': np.sqrt(safe_divide((error**2).sum(axis=(0, 1)), count)).tolist(),
            'bias': safe_divide(error.sum(axis=(0, 1)), count).tolist(),
            'rmse_by_lead': np.sqrt(safe_divide((error**2).sum(axis=0), mask.sum(axis=0))).tolist(),
            'observed_targets_by_lead': mask.sum(axis=0).tolist()}


def wet_scores(probability, arrays):
    mask = arrays['Y_mask'][..., 2]
    p, y = probability[mask], (arrays['Y'][..., 2][mask] >= 1)
    if not len(p):
        return {'observed_targets': 0, 'brier': None, 'reliability': []}
    classified = p >= .5
    hits, misses = int((classified & y).sum()), int((~classified & y).sum())
    false, correct = int((classified & ~y).sum()), int((~classified & ~y).sum())
    bins = []
    indices = np.minimum((p*10).astype(int), 9)
    for i in range(10):
        ok = indices == i
        bins.append({'bin': [i/10, (i+1)/10], 'count': int(ok.sum()),
                     'forecast_probability': float(p[ok].mean()) if ok.any() else None,
                     'observed_frequency': float(y[ok].mean()) if ok.any() else None})
    return {'threshold_mm': 1, 'observed_targets': len(p), 'brier': float(np.mean((p-y)**2)),
            'reliability': bins, 'classification_threshold': .5,
            'hits': hits, 'misses': misses, 'false_alarms': false, 'correct_negatives': correct,
            'probability_of_detection': hits/(hits+misses) if hits+misses else None,
            'false_alarm_ratio': false/(hits+false) if hits+false else None,
            'false_alarm_rate': false/(false+correct) if false+correct else None}


def residual_quantiles(prediction, arrays, coverage=.9):
    quantiles = np.full((14, 3), np.nan, np.float32)
    counts = arrays['Y_mask'].sum(axis=0)
    error = abs(prediction-arrays['Y'])
    for lead in range(14):
        for variable in range(3):
            residual = error[:, lead, variable][arrays['Y_mask'][:, lead, variable]]
            n = len(residual)
            if n >= 10:
                # Finite-sample split-conformal order statistic, without claiming exchangeability.
                rank = min(n, int(np.ceil((n+1)*coverage)))
                quantiles[lead, variable] = np.partition(residual, rank-1)[rank-1]
    return quantiles, counts


def interval_scores(prediction, arrays, quantiles):
    lower, upper = prediction-quantiles, prediction+quantiles
    lower[..., 2] = np.maximum(lower[..., 2], 0)
    mask = arrays['Y_mask'] & np.isfinite(quantiles)
    inside = (arrays['Y'] >= lower) & (arrays['Y'] <= upper)
    count = mask.sum(axis=(0, 1))
    return {'nominal_coverage': .9, 'observed_targets': count.tolist(),
            'empirical_coverage': safe_divide((inside & mask).sum(axis=(0, 1)), count).tolist(),
            'mean_width': safe_divide(np.where(mask, upper-lower, 0).sum(axis=(0, 1)), count).tolist(),
            'coverage_by_lead': safe_divide((inside & mask).sum(axis=0), mask.sum(axis=0)).tolist()}, lower, upper


def paired_skill(prediction, reference, arrays, repeats=400):
    """Two-way station/month cluster bootstrap; few months limit interpretation."""
    mask = arrays['Y_mask']
    loss = np.where(mask, (prediction-arrays['Y'])**2, 0).sum(axis=1).astype(float)
    ref_loss = np.where(mask, (reference-arrays['Y'])**2, 0).sum(axis=1).astype(float)
    _, stations = np.unique(arrays['station_id'], return_inverse=True)
    months, times = np.unique(np.array([str(x)[:7] for x in arrays['target_start']]), return_inverse=True)
    station_count, month_count = stations.max()+1, len(months)
    rng = np.random.default_rng(20261014)
    boot = []
    for _ in range(repeats):
        sw = rng.multinomial(station_count, np.full(station_count, 1/station_count))
        tw = rng.multinomial(month_count, np.full(month_count, 1/month_count))
        weights = sw[stations]*tw[times]
        boot.append(1-safe_divide(weights@loss, weights@ref_loss))
    return {'mse_skill': (1-safe_divide(loss.sum(axis=0), ref_loss.sum(axis=0))).tolist(),
            'cluster_bootstrap_95_percent_interval': np.nanquantile(boot, [.025, .975], axis=0).T.tolist(),
            'station_clusters': int(station_count), 'month_clusters': month_count, 'repeats': repeats,
            'method': 'independent resampling weights over station and target-start calendar-month clusters',
            'limitation': 'Approximate dependence treatment; few audit months and no whole-event/region exclusion; not a forecast prediction interval'}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value
