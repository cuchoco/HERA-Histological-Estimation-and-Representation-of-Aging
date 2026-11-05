"""Unsupervised mean-std alignment of patch embeddings to the GTEx feature distribution.

Each embedding dimension of an external cohort is standardized with the cohort's own moments and
rescaled to the GTEx moments:  x' = (x - mu_cohort) / sd_cohort * sd_gtex + mu_gtex.
Moments are estimated from up to 2,000 randomly drawn patches per slide; no labels are used.
"""
import numpy as np
import torch

from hera.data import read_features

MAX_PATCHES = 2000


def feature_moments(paths, max_patches=MAX_PATCHES, device='cpu'):
    """Per-dimension mean and s.d. over the patches of `paths` (slide i is subsampled with seed i)."""
    n, s, ss = 0, 0.0, 0.0
    for i, p in enumerate(paths):
        x = read_features(p)
        if len(x) > max_patches:
            x = x[np.random.default_rng(i).choice(len(x), max_patches, replace=False)]
        x = x.to(device).double()
        n += x.shape[0]
        s = s + x.sum(0)
        ss = ss + (x * x).sum(0)
    mean = s / n
    sd = torch.sqrt((ss / n - mean * mean).clamp(min=1e-8))
    return mean.float().cpu(), sd.float().cpu()


def align(x, src_mean, src_sd, ref_mean, ref_sd):
    return (x - src_mean) / src_sd * ref_sd + ref_mean
