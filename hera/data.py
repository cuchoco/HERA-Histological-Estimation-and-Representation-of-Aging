"""Slide-level bags of UNI2-h patch embeddings.

One slide = one TRIDENT feature file `<features_dir>/<slide_id>.h5` with dataset 'features' (N, 1536).
Slides in a batch are zero-padded to the longest bag and come with a boolean mask.
"""
from pathlib import Path

import h5py
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from hera.config import AGE_MEAN, AGE_STD


def read_features(path) -> torch.Tensor:
    with h5py.File(path, 'r') as f:
        return torch.from_numpy(f['features'][:]).float()


class SlideDataset(Dataset):
    """rows of `df` need: slide_id, age.  __getitem__ -> (features (N, D), standardized age)"""

    def __init__(self, df: pd.DataFrame, features_dir):
        self.df = df.reset_index(drop=True)
        self.features_dir = Path(features_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = read_features(self.features_dir / f'{r.slide_id}.h5')
        y = torch.tensor((r.age - AGE_MEAN) / AGE_STD, dtype=torch.float32)
        return x, y


def collate(batch):
    """-> padded features (B, Nmax, D), ages (B,), mask (B, Nmax) bool"""
    xs = [b[0] for b in batch]
    x = pad_sequence(xs, batch_first=True, padding_value=0.0)
    lens = torch.tensor([len(t) for t in xs])
    mask = torch.arange(x.shape[1]).expand(len(xs), -1) < lens[:, None]
    return x, torch.stack([b[1] for b in batch]), mask
