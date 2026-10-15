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
report = run(data_root,output,epochs=20,max_seconds=1200)
# Report rendering also stays on the cloud worker.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
methods = ['climatology','ridge','mlp_raw','transformer_raw','champion']
labels = ['Seasonal normal','Ridge','Residual MLP','Residual Transformer','Validation selection']
colors = ['#9ca3af','#3b82f6','#f59e0b','#8b5cf6','#0d9488']
fig,axes = plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
for row,split in enumerate(['audit_2026','station_holdout_audit_2026']):
    for variable,title in enumerate(['Daily high (°C)','Daily low (°C)','Daily rain (mm)']):
        ax = axes[row,variable]
        values = [report['evaluation'][split][m]['point']['rmse'][variable] for m in methods]
        ax.bar(range(len(methods)),values,color=colors)
        ax.set_xticks(range(len(methods)),labels,rotation=35,ha='right',fontsize=8)
        ax.set_title(('Regular stations — ' if row==0 else 'Held-out fit stations — ')+title,fontsize=10)
        ax.set_ylabel('RMSE; lower is better')
        for index,value in enumerate(values):
            ax.text(index,value,f'{value:.3f}',ha='center',va='bottom',fontsize=8)
fig.suptitle('Lilith CONUS: 2026 retrospective NOAA observation audit',fontsize=16)
fig.savefig(output/'audit-rmse.png',dpi=160)
plt.close(fig)
fig,axes = plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
for variable,title in enumerate(['Daily high (°C)','Daily low (°C)','Daily rain (mm)']):
    for method,label,color in zip(methods,labels,colors):
        values = report['evaluation']['audit_2026'][method]['point']['rmse_by_lead']
        axes[variable].plot(range(1,15),[v[variable] for v in values],label=label,color=color)
    axes[variable].set_title(title);axes[variable].set_xlabel('Target lead (days after assumed issue)')
    axes[variable].set_ylabel('RMSE');axes[variable].grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.savefig(output/'audit-lead-rmse.png',dpi=160)
plt.close(fig)
with (output/'audit-summary.csv').open('w',newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['cohort','model','high_rmse_c','low_rmse_c','rain_rmse_mm','wet_brier',
                     'high_interval_coverage','low_interval_coverage','rain_interval_coverage'])
    for split in ['audit_2026','station_holdout_audit_2026']:
        for model,result in report['evaluation'][split].items():
            writer.writerow([split,model,*result['point']['rmse'],result['wet_day']['brier'],
                             *result['interval']['empirical_coverage']])
files = sorted(p for p in output.rglob('*') if p.is_file() and p.name!='CHECKSUMS.sha256')
(output/'CHECKSUMS.sha256').write_text(''.join(
    hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(output).as_posix()+'\n' for p in files))
print('KAGGLE_CONUS_FIT_AND_AUDIT_COMPLETE',flush=True)
