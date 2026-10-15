"""Private Kaggle CUDA fitting/evaluation of the frozen Modal-prepared dataset."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
from pathlib import Path
import hashlib
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone

input_root = Path('/kaggle/input')
source = list(input_root.rglob('train_masked.py'))
base = Path('/kaggle/working/lilith-source')
base.mkdir(exist_ok=True)
if source:
    source_root = source[0].parent.parent
else:
    archives = list(input_root.rglob('source.zip'))
    if len(archives) != 1:
        raise RuntimeError('Expected one immutable source snapshot')
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(base)
    source_root = base
prepared = list(input_root.rglob('train.npz'))
if prepared:
    data_root = prepared[0].parent
else:
    archives = list(input_root.rglob('benchmark-data.zip'))
    if len(archives) != 1:
        raise RuntimeError('Expected the frozen Modal-prepared benchmark')
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(base)
    data_root = base/'data'
sys.path.insert(0,str(source_root))
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Kaggle virtual CUDA GPU is required; refuse CPU model training')
output = Path('/kaggle/working/conus-results')
output.mkdir(exist_ok=True)
execution = {'started_at':datetime.now(timezone.utc).isoformat(), 'provider':'Kaggle',
             'kernel':'codychurchwell/lilith-conus-residual-audit-20261014',
             'python':platform.python_version(), 'torch':torch.__version__, 'cuda':torch.version.cuda,
             'gpu':torch.cuda.get_device_name(0), 'cuda_device_count':torch.cuda.device_count(),
             'data_preparation_provider':'Modal',
             'dataset_manifest_sha256':hashlib.sha256((data_root/'manifest.json').read_bytes()).hexdigest(),
             'source_sha256':{str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((source_root/'training').rglob('*.py'))}}
(output/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
print(json.dumps(execution),flush=True)
from training.train_masked import run
run(data_root,output,epochs=20,max_seconds=1200)
print('KAGGLE_CONUS_FIT_AND_AUDIT_COMPLETE',flush=True)
