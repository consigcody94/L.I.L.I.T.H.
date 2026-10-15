"""Remote NOAA acquisition/preparation/tests; GPU model fitting belongs on Kaggle."""

from pathlib import Path
import re
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/source')
app = modal.App('lilith-conus-preparation')
volume = modal.Volume.from_name('lilith-verified-research',create_if_missing=True)
cpu_image = modal.Image.debian_slim(python_version='3.11').pip_install('numpy==2.2.6')
test_image = modal.Image.debian_slim(python_version='3.11').pip_install('numpy==2.2.6','torch==2.8.0','pytest==9.1.1')
if modal.is_local():
    for label in ['cpu_image','test_image']:
        image = globals()[label]
        image = image.add_local_dir(ROOT/'data','/source/data',ignore=['raw','processed','storage','__pycache__'])
        image = image.add_local_dir(ROOT/'training','/source/training',ignore=['__pycache__','cloud'])
        image = image.add_local_dir(ROOT/'tests','/source/tests',ignore=['__pycache__'])
        image = image.add_local_dir(ROOT/'docs/lilith/experiments/conus-round2','/source/protocol')
        globals()[label] = image


@app.function(image=test_image,cpu=2,memory=4096,timeout=600,max_containers=1,retries=0,
              volumes={'/artifacts':volume})
def test_source(run_name):
    import os,subprocess
    env = {**os.environ,'PYTHONPATH':'/source','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
    result = subprocess.run(['python','-m','pytest','-c','/dev/null','/source/tests/test_masked_training.py','-q'],
                             cwd='/source',env=env,capture_output=True,text=True,timeout=500)
    target = Path('/artifacts/conus-preparation')/run_name
    target.mkdir(parents=True,exist_ok=True)
    (target/'remote_tests.log').write_text(result.stdout+result.stderr)
    volume.commit()
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    print(result.stdout,flush=True)
    return {'remote_new_tests':'passed','execution':'Modal CPU tests, not model fitting'}


@app.function(image=cpu_image,cpu=4,memory=16384,timeout=3600,max_containers=1,retries=0,
              volumes={'/artifacts':volume})
def prepare_remote(run_name):
    import sys,json,hashlib,time,zipfile
    import numpy as np
    sys.path.insert(0,'/source')
    from training.conus_catalog import build
    from training.prepare_masked import prepare
    volume.reload()
    root = Path('/artifacts/conus-preparation')/run_name
    if (root/'PREPARATION_COMPLETE.json').exists():
        raise FileExistsError('Use a new immutable preparation name')
    root.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    if (root/'data/manifest.json').exists():
        manifest = json.loads((root/'data/manifest.json').read_text())
        cohort = json.loads((root/'catalog/CONUS_COHORT.json').read_text())
        print('Reusing completed immutable dataset; validating before publication',flush=True)
    else:
        cohort = build(root/'catalog',Path('/artifacts/cache'))
        manifest = prepare(root/'data',Path('/artifacts/cache'),None,Path('/source/protocol/PROTOCOL.md'),
                           cohort_file=root/'catalog/CONUS_COHORT.json')
    # Check every observed target against the retained source calendar, never imputed truth.
    with np.load(root/'data/station_calendars.npz',allow_pickle=False) as c:
        calendars,ids = c['values'],c['station_id']
        date0 = np.datetime64(c['dates'][0])
    lookup = {str(s):i for i,s in enumerate(ids)}
    support = {s['id']:s['normal_support'] for s in manifest['stations']}
    validation = {}
    for name,expected in manifest['split_sha256'].items():
        file = root/'data'/f'{name}.npz'
        assert hashlib.sha256(file.read_bytes()).hexdigest() == expected
        with np.load(file,allow_pickle=False) as arrays:
            idx = np.array([lookup[str(s)] for s in arrays['station_id']])
            start = (arrays['target_start'].astype('datetime64[D]')-date0).astype(int)
            actual = calendars[idx[:,None],start[:,None]+np.arange(14)]
            mask = np.isfinite(actual)&np.array([support[str(s)] for s in arrays['station_id']])[:,None,:]
            np.testing.assert_array_equal(arrays['Y_mask'],mask)
            np.testing.assert_array_equal(arrays['Y'][mask],actual[mask])
            for key in ['X','Y','climatology','X_climatology','wet_climatology']:
                assert np.isfinite(arrays[key]).all(),(name,key)
            validation[name] = {'cases':len(idx),'stations':len(set(arrays['station_id'])),
                                'source_calendar_truth_exact':True,'observed_targets':mask.sum(axis=(0,1)).tolist(),
                                'first_target':str(min(arrays['target_start'])),
                                'last_target_end':str(max(arrays['target_start'].astype('datetime64[D]'))+np.timedelta64(13,'D'))}
    (root/'DATA_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
    volume.commit()
    # Package on Modal; do not recompress scientific archives on the Mac.
    with zipfile.ZipFile(root/'benchmark-data.zip','w',compression=zipfile.ZIP_STORED,strict_timestamps=False) as archive:
        for path in sorted((root/'data').iterdir()):
            archive.write(path,'data/'+path.name)
    with zipfile.ZipFile(root/'noaa-source-snapshots.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=2,strict_timestamps=False) as archive:
        for station in manifest['stations']:
            archive.write(Path('/artifacts/cache/all')/(station['id']+'.dly'),'all/'+station['id']+'.dly')
        for name in ['ghcnd-stations.txt','ghcnd-inventory.txt']:
            archive.write(Path('/artifacts/cache')/name,name)
        for path in sorted((root/'catalog').iterdir()):
            archive.write(path,'catalog/'+path.name)
    source_files = []
    for directory in ['training','data','tests']:
        source_files.extend(p for p in (Path('/source')/directory).rglob('*.py') if '__pycache__' not in str(p))
    with zipfile.ZipFile(root/'source.zip','w',compression=zipfile.ZIP_DEFLATED,strict_timestamps=False) as archive:
        for file in sorted(source_files):
            archive.write(file,str(file.relative_to('/source')))
        archive.write('/source/protocol/PROTOCOL.md','PROTOCOL.md')
    package_names = ['benchmark-data.zip','noaa-source-snapshots.zip','source.zip','DATA_VALIDATION.json','remote_tests.log']
    result = {'scope':'CONUS','run_name':run_name,'station_count':len(manifest['stations']),
              'registry_counts':cohort['registry_counts_by_network'],'unmatched_registry_count':cohort['unmatched_count'],
              'counts':manifest['counts'],'elapsed_seconds':round(time.monotonic()-started,1),
              'preparation_environment':'Modal 4 CPU, 16 GiB; no local NOAA processing or training',
              'package_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in package_names}}
    (root/'PREPARATION_COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
    export = root/'export';export.mkdir(exist_ok=True)
    import shutil
    for name in package_names+['PREPARATION_COMPLETE.json']:
        shutil.copyfile(root/name,export/name)
    volume.commit()
    print(json.dumps(result),flush=True)
    return result


@app.local_entrypoint()
def main(run_name:str='conus-asos-awos-20261014-v1'):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',run_name):
        raise ValueError('Invalid immutable run name')
    print(test_source.remote(run_name))
    print(prepare_remote.remote(run_name))
