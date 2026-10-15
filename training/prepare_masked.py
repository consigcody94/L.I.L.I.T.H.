"""Masked station benchmark with frozen tuning, calibration and 2026 audit years."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from data.providers.ghcn_records import BASE_URL, decode_daily, download, station_metadata

FEATURES = ("TMAX", "TMIN", "PRCP")
SPLITS = {"train": (1980, 2016), "validation": (2017, 2020),
          "calibration": (2021, 2023), "development": (2024, 2025), "audit_2026": (2026, 2026)}


def calendar_bin(day: date) -> int:
    """Stable month/day seasonality: March does not shift between leap/nonleap years."""
    return date(2000, day.month, day.day).timetuple().tm_yday - 1


def split_for_window(first: date, last: date) -> str | None:
    for name, (start, end) in SPLITS.items():
        if first >= date(start, 1, 1) and last <= date(end, 12, 31):
            return name
    return None


def seasonal_normals(values, bins, train_mask):
    """31-day circular means, fit on training years; unsupported targets stay masked."""
    normals = np.zeros((366, 3), np.float32)
    wet = np.zeros(366, np.float32)
    support = np.zeros(3, bool)
    fallback_bins = []
    for variable in range(3):
        ok = train_mask & np.isfinite(values[:, variable])
        support[variable] = ok.sum() >= 100
        if not support[variable]:
            fallback_bins.append(366)
            continue
        counts = np.bincount(bins[ok], minlength=366)
        sums = np.bincount(bins[ok], weights=values[ok, variable], minlength=366)
        window_count = sum(np.roll(counts, shift) for shift in range(-15, 16))
        window_sum = sum(np.roll(sums, shift) for shift in range(-15, 16))
        fallback = values[ok, variable].mean()
        normals[:, variable] = np.divide(window_sum, window_count,
                                         out=np.full(366, fallback, dtype=float), where=window_count >= 5)
        fallback_bins.append(int((window_count < 5).sum()))
        if variable == 2:
            wet_sum = np.bincount(bins[ok], weights=values[ok, variable] >= 1, minlength=366)
            wet_sum = sum(np.roll(wet_sum, shift) for shift in range(-15, 16))
            wet[:] = np.divide(wet_sum, window_count, out=np.full(366, (values[ok, 2] >= 1).mean()),
                               where=window_count >= 5)
    return normals, wet, support, fallback_bins


def choose_cohort(metadata, inventory, old_manifest, limit):
    coverage = defaultdict(set)
    for line in inventory.splitlines():
        parts = line.split()
        if len(parts) == 6:
            sid, _, _, element, first, last = parts
            if element in FEATURES and int(first) <= 2000 and int(last) >= 2025:
                coverage[sid].add(element)
    selected = [dict(metadata[s['id']]) for s in old_manifest['stations'] if s['id'] in metadata][:limit]
    used = {s['id'] for s in selected}
    cells = defaultdict(list)
    for sid, elements in coverage.items():
        if sid in used or sid not in metadata or len(elements) < 2 or not elements.intersection({'TMAX', 'TMIN'}):
            continue
        station = metadata[sid]
        cell = (min(5, int((station['latitude'] + 90) // 30)), min(11, int((station['longitude'] + 180) // 30)))
        cells[cell].append(station)
    for group in cells.values():
        group.sort(key=lambda s: hashlib.sha256(s['id'].encode()).hexdigest())
    depth = 0
    while len(selected) < limit:
        added = False
        for cell in sorted(cells):
            if depth < len(cells[cell]) and len(selected) < limit:
                selected.append(dict(cells[cell][depth])); added = True
        if not added:
            break
        depth += 1
    for station in selected:
        station['station_holdout'] = int(hashlib.sha256(station['id'].encode()).hexdigest()[:8], 16) % 5 == 0
    return selected


def prepare(output: Path, cache: Path, old_manifest: Path | None, protocol: Path, limit=288,
            max_train=120000, cohort_file: Path | None = None):
    if (output / 'manifest.json').exists():
        raise FileExistsError('Use a new snapshot directory')
    output.mkdir(parents=True, exist_ok=True)
    catalog, catalog_hash = download('ghcnd-stations.txt', cache)
    inventory, inventory_hash = download('ghcnd-inventory.txt', cache)
    cohort = json.loads(cohort_file.read_text()) if cohort_file else None
    stations = cohort['stations'] if cohort else choose_cohort(
        station_metadata(catalog), inventory, json.loads(old_manifest.read_text()), limit)
    first, last = date(1980, 1, 1), date(2026, 9, 30)
    days = [first + timedelta(days=i) for i in range((last-first).days+1)]
    day_indices = {d.isoformat(): i for i, d in enumerate(days)}
    bins = np.array([calendar_bin(d) for d in days])
    fit_mask = np.array([d.year <= 2016 for d in days])
    manifests, failures, calendars, normals_all, wet_all, cases = [], [], [], [], [], []

    def retrieve(station):
        try:
            text, digest = download(f'all/{station["id"]}.dly', cache)
            return station, text, digest, None
        except OSError as exc:
            return station, '', '', type(exc).__name__

    # Bounded four-way archive retrieval, never an unbounded API burst.
    with ThreadPoolExecutor(max_workers=4) as executor:
        for number, (station, text, digest, failure) in enumerate(executor.map(retrieve, stations)):
            sid = station['id']
            if failure:
                failures.append({'station_id': sid, 'reason': failure}); continue
            values = np.full((len(days), 3), np.nan, np.float32)
            subset = '\n'.join(line for line in text.splitlines()
                               if len(line) >= 269 and line[17:21] in FEATURES and '1980' <= line[11:15] <= '2026')
            for row in decode_daily(subset, sid):
                index = day_indices.get(row['date'])
                if index is not None and row['value'] is not None:
                    values[index, FEATURES.index(row['element'])] = row['value']
            normal, wet, support, fallback = seasonal_normals(values, bins, fit_mask)
            index = len(manifests)
            counts = defaultdict(int)
            # Anchor independently within each partition; targets never overlap.
            for split, (year0, year1) in SPLITS.items():
                if station['station_holdout'] and split in {'train', 'validation', 'calibration'}:
                    continue
                begin = day_indices[date(year0, 1, 1).isoformat()] + 32
                end_day = min(last, date(year1, 12, 31))
                end = day_indices[end_day.isoformat()] - 13
                for target in range(begin, end + 1, 14):
                    x = values[target-32:target-2]
                    y = values[target:target+14]
                    ymask = np.isfinite(y) & support
                    if np.isfinite(x).sum() < 18 or not ymask.any():
                        counts['excluded_windows'] += 1; continue
                    name = ('station_holdout_' + split) if station['station_holdout'] else split
                    cases.append((index, target, name)); counts[name] += 1
            entry = {**station, 'url': BASE_URL+f'all/{sid}.dly', 'sha256': digest,
                     'observed_elements_by_variable': np.isfinite(values).sum(axis=0).tolist(),
                     'normal_support': support.tolist(), 'normal_annual_fallback_bins': fallback,
                     'window_counts': dict(counts)}
            manifests.append(entry); calendars.append(values); normals_all.append(normal); wet_all.append(wet)
            print(json.dumps({'station': number+1, 'id': sid, 'windows': dict(counts)}), flush=True)

    rng = np.random.default_rng(20261014)
    train_by_station = defaultdict(list)
    other_cases = []
    for case in cases:
        (train_by_station[case[0]] if case[2] == 'train' else other_cases).append(case)
    for group in train_by_station.values():
        rng.shuffle(group)
    training = []
    depth = 0
    while len(training) < max_train:
        added = False
        for station_index in sorted(train_by_station):
            group = train_by_station[station_index]
            if depth < len(group) and len(training) < max_train:
                training.append(group[depth]); added = True
        if not added:
            break
        depth += 1
    cases = training + other_cases
    by_split = defaultdict(list)
    for case in cases:
        by_split[case[2]].append(case)
    hashes, counts = {}, {}
    for name, group in by_split.items():
        rows = defaultdict(list)
        for station_index, target, _ in group:
            station, values = manifests[station_index], calendars[station_index]
            x, y = values[target-32:target-2].copy(), values[target:target+14].copy()
            xc, yc = normals_all[station_index][bins[target-32:target-2]], normals_all[station_index][bins[target:target+14]]
            xmask, ymask = np.isfinite(x), np.isfinite(y) & station['normal_support']
            rows['X'].append(np.where(xmask, x, xc)); rows['X_mask'].append(xmask)
            rows['Y'].append(np.where(ymask, y, 0)); rows['Y_mask'].append(ymask)
            rows['X_climatology'].append(xc); rows['climatology'].append(yc)
            rows['wet_climatology'].append(wet_all[station_index][bins[target:target+14]])
            angle = 2*np.pi*bins[target]/366
            rows['meta'].append([station['latitude']/90, station['longitude']/180,
                                 max(-500, station['elevation'])/5000, np.sin(angle), np.cos(angle)])
            rows['station_id'].append(station['id']); rows['target_start'].append(days[target].isoformat())
            rows['station_holdout'].append(station['station_holdout'])
        arrays = {key: np.asarray(value, dtype=(bool if key in {'X_mask','Y_mask','station_holdout'} else
                    str if key in {'station_id','target_start'} else np.float32)) for key,value in rows.items()}
        np.savez_compressed(output/f'{name}.npz', **arrays)
        counts[name] = {'cases': len(group), 'stations': len(set(arrays['station_id'])),
                        'observed_targets': arrays['Y_mask'].sum(axis=(0,1)).tolist()}
        hashes[name] = hashlib.sha256((output/f'{name}.npz').read_bytes()).hexdigest()
    np.savez_compressed(output/'station_calendars.npz', values=np.asarray(calendars),
                        normals=np.asarray(normals_all), wet_normals=np.asarray(wet_all),
                        station_id=np.array([s['id'] for s in manifests]), dates=np.array([d.isoformat() for d in days]))
    manifest = {'schema_version': 2, 'created_at': datetime.now(timezone.utc).isoformat(),
                'features': list(FEATURES), 'units': ['degC','degC','mm'], 'input_days':30, 'target_days':14,
                'gap_days':2, 'stride_days':14, 'source':'measured NOAA GHCN-Daily, QC accepted; station reporting days',
                'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
                'catalog_sha256':catalog_hash, 'inventory_sha256':inventory_hash, 'stations':manifests,
                'download_failures':failures, 'counts':counts, 'split_sha256':hashes,
                'split_policy':SPLITS, 'training_cases_before_cap':sum(map(len,train_by_station.values())),
                'max_train_cases':max_train,
                'scope':cohort.get('scope','global_land_sample') if cohort else 'global_land_sample',
                'cohort_sha256':hashlib.sha256(cohort_file.read_bytes()).hexdigest() if cohort_file else None,
                'station_sampling':cohort.get('selection_policy') if cohort else 'old144 plus hash-ordered 30-degree-cell candidates',
                'normal_policy':'1980-2016 own-station 31-day seasonal means, leap-aligned; annual fallback for <5 bin samples; >=100 historical observations required per variable',
                'limitations':['Retrospective final historical revisions; original availability unknown; two-day embargo is assumed, not verified latency',
                               'Land stations only; no NWP, ocean, wind or severe-event skill claim',
                               'Station holdout permits local historical climatology; not zero-history or region holdout',
                               'Held-out-station intervals use calibration residuals transferred from fitting stations',
                               'Partial targets stay masked; input imputation is train-only normal plus explicit mask',
                               '2024-2025 development years previously inspected; 2026 audit is retrospective, not prospective']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'PROTOCOL.md').write_text(protocol.read_text())
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--old-manifest', type=Path)
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--stations', type=int, default=288)
    args = parser.parse_args()
    if not args.cohort and not args.old_manifest:
        parser.error('Provide --cohort or --old-manifest')
    print(json.dumps(prepare(args.output,args.cache,args.old_manifest,args.protocol,args.stations,
                             cohort_file=args.cohort)['counts']))
