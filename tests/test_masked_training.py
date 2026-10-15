"""Leakage, missing-target and residual-model contracts for the Kaggle benchmark."""

from datetime import date

import numpy as np
import pytest
import torch

from training.prepare_masked import calendar_bin, seasonal_normals, split_for_window
from training.masked_models import ResidualStationModel, anomaly_features, masked_objective
from training.masked_metrics import interval_scores, json_safe, residual_quantiles, scores, wet_scores
from training.train_masked import damped_persistence, select_blend


@pytest.fixture(scope='module', autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def example(n=20):
    return {'X':np.ones((n,30,3),np.float32)*5, 'X_climatology':np.ones((n,30,3),np.float32)*5,
            'X_mask':np.ones((n,30,3),bool), 'Y':np.ones((n,14,3),np.float32)*5,
            'Y_mask':np.ones((n,14,3),bool), 'climatology':np.ones((n,14,3),np.float32)*5,
            'wet_climatology':np.ones((n,14),np.float32)*.3,
            'meta':np.zeros((n,5),np.float32), 'station_id':np.array(['TEST0000001']*n),
            'target_start':np.array(['2026-02-02']*n)}


def test_leap_alignment_and_split_purges():
    assert calendar_bin(date(2023,3,1)) == calendar_bin(date(2024,3,1))
    assert calendar_bin(date(2024,2,29))+1 == calendar_bin(date(2023,3,1))
    assert split_for_window(date(2020,12,31),date(2021,2,14)) is None
    assert split_for_window(date(2021,1,1),date(2023,12,31)) == 'calibration'
    assert split_for_window(date(2026,1,1),date(2026,9,30)) == 'audit_2026'


def test_future_observations_cannot_change_training_normals():
    values = np.ones((732,3),np.float32)
    bins = np.tile(np.arange(366),2)
    train = np.arange(732)<366
    before = seasonal_normals(values,bins,train)
    values[~train] = 10000
    after = seasonal_normals(values,bins,train)
    np.testing.assert_array_equal(before[0],after[0])
    np.testing.assert_array_equal(before[1],after[1])


def test_features_use_each_inputs_own_normal_and_explicit_masks():
    arrays = example()
    arrays['X_climatology'][:] = np.arange(30)[None,:,None]
    arrays['X'][:] = arrays['X_climatology']
    arrays['climatology'][:] = 500
    # First mean-anomaly block is zero even when target seasonal normals differ.
    np.testing.assert_array_equal(anomaly_features(arrays)[:,:3],0)
    arrays['X_mask'][:,-1] = False
    arrays['X'][:,-1] = 1e9
    np.testing.assert_array_equal(anomaly_features(arrays)[:,:3],0)
    np.testing.assert_allclose(damped_persistence(arrays),arrays['climatology'])


def test_masked_truth_never_enters_scores_or_training_loss():
    arrays = example()
    arrays['Y_mask'][:,0,2] = False
    prediction = arrays['Y'].copy()
    first = scores(prediction,arrays)
    wet_first = wet_scores(arrays['wet_climatology'],arrays)
    point = torch.tensor(prediction,requires_grad=True)
    logits = torch.zeros(20,14,requires_grad=True)
    truth, mask = torch.tensor(arrays['Y']),torch.tensor(arrays['Y_mask'])
    a = masked_objective(point,logits,truth,mask,torch.ones(3))
    arrays['Y'][:,0,2] = 1e9
    b = masked_objective(point,logits,torch.tensor(arrays['Y']),mask,torch.ones(3))
    assert a.item() == b.item()
    assert json_safe(scores(prediction,arrays)) == json_safe(first)
    assert wet_scores(arrays['wet_climatology'],arrays) == wet_first
    b.backward()
    assert torch.isfinite(point.grad).all()
    assert (point.grad[:,0,2] == 0).all()


def test_interval_calibration_uses_only_observed_residuals():
    arrays = example()
    prediction = arrays['Y']+2
    arrays['Y_mask'][:,0,2] = False
    arrays['Y'][:,0,2] = 1e8
    quantiles, counts = residual_quantiles(prediction,arrays)
    assert np.isnan(quantiles[0,2]) and counts[0,2] == 0
    assert quantiles[1,2] == 2
    result,lo,hi = interval_scores(prediction,arrays,quantiles)
    assert result['observed_targets'][2] == 20*13
    assert result['empirical_coverage'] == [1.,1.,1.]
    assert np.all(lo[:,1:,2] >= 0)
    assert np.all(hi[:,1:,2] >= lo[:,1:,2])


def test_validation_can_reject_neural_blend_entirely():
    arrays = example()
    weights,pw = select_blend(arrays['Y']+10,np.zeros((20,14)),arrays,arrays['Y'])
    np.testing.assert_array_equal(weights,0)
    assert pw == 0


@pytest.mark.parametrize('kind',['mlp','transformer'])
def test_neural_residual_shapes_physical_outputs_and_finite_gradient(kind):
    model = ResidualStationModel(kind,45,[4,4,8])
    history = torch.zeros(2,30,9); history[:,:,3:6] = 1
    history[:,0,3:6] = 0
    future = torch.zeros(2,14,7); future[:,:,6] = .3
    base = torch.tensor([10.,5.,0.]).expand(2,14,3)
    point,logits = model(history,torch.zeros(2,45),future,base)
    assert point.shape == (2,14,3) and logits.shape == (2,14)
    assert (point[:,:,0] >= point[:,:,1]).all() and (point[:,:,2] >= 0).all()
    loss = masked_objective(point,logits,torch.ones_like(point),torch.ones_like(point,dtype=torch.bool),model.scales)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
