from pathlib import Path
import sys, zipfile
base = Path('/kaggle/working/lilith')
base.mkdir(exist_ok=True)
input_root = Path('/kaggle/input')
source_candidates = list(input_root.rglob('train_verified.py'))
if source_candidates:
    source_root = source_candidates[0].parent.parent
else:
    archives = list(input_root.rglob('source.zip'))
    if len(archives) != 1:
        raise RuntimeError('Source archive or extracted training module is unavailable')
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(base)
    source_root = base
train_candidates = list(input_root.rglob('train.npz'))
if train_candidates:
    data_root = train_candidates[0].parent
else:
    archives = list(input_root.rglob('benchmark-data.zip'))
    if len(archives) != 1:
        raise RuntimeError('Frozen benchmark data is unavailable')
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(base)
    data_root = base / 'data'
sys.path.insert(0, str(source_root))
import torch
if not torch.cuda.is_available():
    raise RuntimeError('GPU required for this independent reproduction')
from training.train_verified import train
train(data_root, Path('/kaggle/working/reproduction'), epochs=12, max_seconds=1200)
