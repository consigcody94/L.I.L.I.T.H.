"""Fixed Kaggle experiment: fit/tune, freeze selection, calibrate, then audit NOAA truth."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.masked_models import (ResidualStationModel, anomaly_features, constrain,
                                    fit_ridge, masked_objective, neural_inputs, ridge_predict)
from training.masked_metrics import (interval_scores, json_safe, paired_skill, residual_quantiles,
                                     scores, wet_scores)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False)+'\n')


def load_split(data, name):
    with np.load(data/f'{name}.npz', allow_pickle=False) as loaded:
        return dict(loaded)


def damped_persistence(arrays):
    mask = arrays['X_mask']
    anomaly = np.where(mask, arrays['X']-arrays['X_climatology'], 0)
    age = mask[:, ::-1].argmax(axis=1)
    last = np.take_along_axis(anomaly, (29-age)[:, None, :], axis=1)[:, 0]
    last = np.where(mask.any(axis=1), last, 0)
    # Includes the two-day gap and actual age of the last measured variable.
    decay = np.exp(-(np.arange(1,15)[None,:,None]+2+age[:,None,:])/7)
    return constrain(arrays['climatology']+last[:,None,:]*decay)


def fit_neural(kind, train, validation, ridge, output, epochs=20, max_seconds=1200, seed=20261014):
    started = time.monotonic()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high'); torch.set_num_threads(4)
    observed = train['X_mask']
    anomaly = train['X']-train['X_climatology']
    scales = np.sqrt(np.where(observed, anomaly**2, 0).sum(axis=(0,1))/observed.sum(axis=(0,1))).clip(.1).astype(np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    model = ResidualStationModel(kind, anomaly_features(train).shape[1], scales).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not bf16)
    def loader(arrays, shuffle=False):
        tensors = [torch.from_numpy(x) for x in neural_inputs(arrays, ridge, scales)]
        tensors += [torch.from_numpy(arrays['Y']), torch.from_numpy(arrays['Y_mask'])]
        return DataLoader(TensorDataset(*tensors), batch_size=512, shuffle=shuffle, num_workers=0,
                          pin_memory=device.type == 'cuda')
    train_loader, val_loader = loader(train, True), loader(validation)
    def predict_loader(loader):
        model.eval(); point, probability = [], []
        with torch.no_grad():
            for batch in loader:
                with torch.autocast(device.type, dtype=torch.bfloat16 if bf16 else torch.float16,
                                    enabled=device.type == 'cuda'):
                    p, logits = model(*(x.to(device) for x in batch[:4]))
                point.append(p.float().cpu().numpy()); probability.append(logits.float().sigmoid().cpu().numpy())
        return np.concatenate(point), np.concatenate(probability)
    best, selected_epoch, history, stale = float('inf'), 0, [], 0
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train(); total, count = 0., 0
        for batch in train_loader:
            batch = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16 if bf16 else torch.float16,
                                enabled=device.type == 'cuda'):
                p, logits = model(*batch[:4])
                loss = masked_objective(p.float(), logits.float(), batch[4], batch[5], model.scales)
            if not torch.isfinite(loss):
                raise RuntimeError(f'{kind}: nonfinite loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer); scaler.update()
            total += loss.item()*len(batch[0]); count += len(batch[0])
        p, probability = predict_loader(val_loader)
        mask = validation['Y_mask']
        error = np.where(mask, ((p-validation['Y'])/scales)**2, 0)
        normalized_mse = error.sum()/mask.sum()
        brier = wet_scores(probability, validation)['brier']
        # The training objective uses BCE, but model selection uses a proper Brier score.
        objective = float(normalized_mse + .2*brier)
        record = {'epoch':epoch+1, 'training_objective':total/count, 'validation_objective':objective,
                  'validation_normalized_mse':float(normalized_mse), 'validation_wet_brier':brier,
                  'elapsed_seconds':round(time.monotonic()-started,1)}
        history.append(record); print(json.dumps({'model':kind, **record}), flush=True)
        if objective < best:
            best, selected_epoch, stale = objective, epoch+1, 0
            torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'kind':kind, 'feature_count':anomaly_features(train).shape[1], 'scales':scales.tolist(),
                        'seed':seed, 'epoch':selected_epoch, 'status':'research_not_promoted'}, output/'candidate.pt')
        else:
            stale += 1
        if stale >= 4 or time.monotonic()-started >= max_seconds:
            break
    model.load_state_dict(torch.load(output/'candidate.pt', map_location=device, weights_only=True)['model_state_dict'])
    report = {'model':kind, 'selected_epoch':selected_epoch, 'seed':seed, 'history':history,
              'hardware':torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
              'precision':'BF16' if bf16 else 'FP16' if device.type == 'cuda' else 'FP32',
              'torch_version':torch.__version__, 'numpy_version':np.__version__,
              'parameters':sum(p.numel() for p in model.parameters()), 'checkpoint_sha256':digest(output/'candidate.pt'),
              'elapsed_seconds':round(time.monotonic()-started,1)}
    save_json(output/'fit_report.json',report)
    return lambda arrays: predict_loader(loader(arrays)), report


def select_blend(raw, probability, arrays, ridge_point):
    mask = arrays['Y_mask']
    trials, probabilities = [], []
    for weight in (0., .25, .5, .75, 1.):
        p = constrain(weight*raw+(1-weight)*ridge_point)
        error = np.where(mask, (p-arrays['Y'])**2, 0)
        trials.append(error.sum(axis=(0,1))/mask.sum(axis=(0,1)))
        probabilities.append(wet_scores(weight*probability+(1-weight)*arrays['wet_climatology'],arrays)['brier'])
    weights = np.array([0.,.25,.5,.75,1.])
    return weights[np.argmin(trials,axis=0)], float(weights[np.argmin(probabilities)])


def run(data: Path, output: Path, epochs=20, max_seconds=1200):
    if (output/'report.json').exists():
        raise FileExistsError('Never overwrite a completed experiment')
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data/'manifest.json').read_text())
    for name, expected in manifest['split_sha256'].items():
        if digest(data/f'{name}.npz') != expected:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    train, validation = load_split(data,'train'), load_split(data,'validation')
    print(json.dumps({'phase':'fit_ridge','training_cases':len(train['X'])}),flush=True)
    ridge = fit_ridge(train, validation)
    np.savez_compressed(output/'ridge.npz', **{k:v for k,v in ridge.items() if k != 'trials'})
    models, fit_reports, val_predictions, blend = {}, {}, {}, {}
    ridge_val = ridge_predict(validation,ridge)
    for kind in ('mlp','transformer'):
        models[kind], fit_reports[kind] = fit_neural(kind,train,validation,ridge,output/kind,epochs,max_seconds)
        val_predictions[kind] = models[kind](validation)
        point, probability = val_predictions[kind]
        w, pw = select_blend(point,probability,validation,ridge_val)
        blend[kind] = {'point_weights':w.tolist(),'wet_probability_weight':pw}

    def predictions(arrays, raw=None):
        base = ridge_predict(arrays,ridge)
        result = {'climatology':(constrain(arrays['climatology']),arrays['wet_climatology']),
                  'damped_persistence':(damped_persistence(arrays),arrays['wet_climatology']),
                  'ridge':(base,arrays['wet_climatology'])}
        for kind in ('mlp','transformer'):
            point, probability = raw[kind] if raw is not None else models[kind](arrays)
            result[kind+'_raw'] = point,probability
            weights = np.array(blend[kind]['point_weights'])
            pw = blend[kind]['wet_probability_weight']
            result[kind+'_blend'] = (constrain(weights*point+(1-weights)*base),
                                    pw*probability+(1-pw)*arrays['wet_climatology'])
        return result

    val = predictions(validation,val_predictions)
    candidates = ['ridge','mlp_blend','transformer_blend']
    val_mse = np.array([np.array(scores(val[k][0],validation)['rmse'])**2 for k in candidates])
    selected = [candidates[i] for i in np.argmin(val_mse,axis=0)]
    wet_selected = min(candidates,key=lambda k:wet_scores(val[k][1],validation)['brier'])
    selection = {'point_model_by_variable':selected,'wet_probability_model':wet_selected,
                 'blends':blend,'ridge_trials':ridge['trials'],'ridge_selected_alphas':ridge['selected_alphas'].tolist(),
                 'validation_mse':{k:val_mse[i].tolist() for i,k in enumerate(candidates)},
                 'selection_data':'validation 2017-2020 only',
                 'frozen_before_loading_calibration_or_audit':True}
    save_json(output/'selection_frozen.json',selection)
    print(json.dumps({'phase':'selection_frozen', **selection}),flush=True)
    def add_champion(preds):
        preds['champion'] = (constrain(np.stack([preds[selected[v]][0][...,v] for v in range(3)],axis=-1)),
                             preds[wet_selected][1])
        return preds
    calibration = load_split(data,'calibration')
    calibration_predictions = add_champion(predictions(calibration))
    intervals = {name:residual_quantiles(p[0],calibration) for name,p in calibration_predictions.items()}
    np.savez_compressed(output/'interval_calibration.npz', **{
        name:values[0] for name,values in intervals.items()}, observed_counts=calibration['Y_mask'].sum(axis=0))
    save_json(output/'calibration.json',{'years':'2021-2023','method':'90% absolute-residual split conformal, per lead/variable',
                                      'assumptions':'weather dependence/shift violates exchangeability; report empirical coverage',
                                      'quantiles':{k:v[0] for k,v in intervals.items()},
                                      'counts':calibration['Y_mask'].sum(axis=0)})
    evaluation = {}
    station_lat = {s['id']:s['latitude'] for s in manifest['stations']}
    evaluation_names = ['validation','calibration','development','station_holdout_development',
                        'audit_2026','station_holdout_audit_2026']
    for split in evaluation_names:
        arrays = validation if split == 'validation' else calibration if split == 'calibration' else load_split(data,split)
        preds = add_champion(val.copy()) if split == 'validation' else calibration_predictions if split == 'calibration' else add_champion(predictions(arrays))
        np.savez_compressed(output/f'{split}_truth.npz', actual=arrays['Y'], observed_mask=arrays['Y_mask'],
                            station_id=arrays['station_id'], target_start=arrays['target_start'],
                            climatology=arrays['climatology'], X_observed_mask=arrays['X_mask'])
        summary = {}
        latitudes = np.array([station_lat[s] for s in arrays['station_id']])
        months = np.array([int(d[5:7]) for d in arrays['target_start']])
        groups = {f'latitude_{lo}_{hi}':(latitudes>=lo)&(latitudes<hi)
                  for lo,hi in [(-90,-30),(-30,0),(0,30),(30,90.01)]}
        groups.update({season:np.isin(months,m) for season,m in
                       {'target_start_DJF':[12,1,2], 'target_start_MAM':[3,4,5],
                        'target_start_JJA':[6,7,8], 'target_start_SON':[9,10,11]}.items()})
        for name,(point,probability) in preds.items():
            interval,lower,upper = interval_scores(point,arrays,intervals[name][0])
            result = {'point':scores(point,arrays),'wet_day':wet_scores(probability,arrays),'interval':interval,
                      'physical_violations':{'negative_rain_fraction':float((point[...,2]<0).mean()),
                                             'high_below_low_fraction':float((point[...,0]<point[...,1]).mean())},
                      'groups':{},'stations':{}}
            for group,ok in groups.items():
                if ok.any():
                    result['groups'][group] = scores(point[ok],{k:v[ok] for k,v in arrays.items()})
            for station in sorted(set(arrays['station_id'])):
                ok = arrays['station_id'] == station
                result['stations'][station] = scores(point[ok],{k:v[ok] for k,v in arrays.items()})
            if split not in {'validation','calibration'}:
                result['skill_vs_climatology'] = paired_skill(point,preds['climatology'][0],arrays)
                result['skill_vs_ridge'] = paired_skill(point,preds['ridge'][0],arrays)
            summary[name] = result
            np.savez_compressed(output/f'{split}_{name}_predictions.npz',prediction=point,wet_probability=probability,
                                lower_90=lower,upper_90=upper)
        evaluation[split] = summary
        print(json.dumps({'phase':'evaluation','split':split,'champion_rmse':summary['champion']['point']['rmse'],
                          'climatology_rmse':summary['climatology']['point']['rmse'],
                          'ridge_rmse':summary['ridge']['point']['rmse']}),flush=True)
    report = {'created_at':datetime.now(timezone.utc).isoformat(),'status':'research_not_promoted',
              'dataset_manifest_sha256':digest(data/'manifest.json'), 'protocol_sha256':manifest['protocol_sha256'],
              'source_type':'measured NOAA GHCN-Daily with per-variable observed target masks',
              'models':fit_reports,'selection':selection,'evaluation':evaluation,
              'limitations':manifest['limitations']+[
                  'One seed per neural architecture; no production promotion or issue-matched NWP baseline',
                  '2026 audit is January-September only; reporting coverage changes substantially',
                  'Intervals are marginal empirical checks; no guaranteed weather exchangeability or simultaneous trajectory coverage',
                  'Wet-day probabilities are not severe-weather warnings; no massive-event capability established',
                  'Prior model absolute scores use different cases and cannot be directly ranked against this cohort'],
              'promotion':{'approved':False,'reason':'Requires prospective availability-aware NWP comparison, regional/event qualification and reliability review'}}
    save_json(output/'report.json',report)
    save_json(output/'manifest.json',manifest)
    files = sorted(p for p in output.rglob('*') if p.is_file() and p.name!='CHECKSUMS.sha256')
    (output/'CHECKSUMS.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(output).as_posix()}\n' for p in files))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--max-seconds',type=int,default=1200)
    args = parser.parse_args()
    run(args.data,args.output,args.epochs,args.max_seconds)
