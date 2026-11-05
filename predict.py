"""Predict histological age for new slides with the fused clock.

Each slide is scored by the five fold models of the pan-organ clock and of the matching per-organ
clock (fold predictions are averaged), and the two are combined with the organ's fusion weight w_o
from fuse.py: fused = (1 - w_o) * pan + w_o * per_organ. Organs without a per-organ clock use
the pan-organ clock alone.

    # every <slide_id>.h5 in a folder, one organ
    python predict.py --features_dir ext/features_uni_v2 --organ Lung --out lung_pred.csv

    # a slide list with columns slide_id, organ and (optionally) age
    python predict.py --features_dir ext/features_uni_v2 --slides ext/slides.csv --out pred.csv

    # with unsupervised mean-std alignment of the embeddings to GTEx (recommended for other hospitals)
    python predict.py ... --align

Features must come from the same TRIDENT / UNI2-h pipeline (scripts/extract_features.sh).
If an `age` column is given, MAE / Pearson r are printed together with a 10-slide recalibration
(slope and intercept fitted on 10 random labelled slides, MAE on the rest, 2,000 random splits).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hera.align import align, feature_moments
from hera.config import AGE_MEAN, AGE_STD, PER_ORGAN_CLOCKS
from hera.data import read_features
from hera.model import load_clock

N_FOLDS = 5


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features_dir', required=True)
    ap.add_argument('--organ', help='organ of every slide in --features_dir (e.g. Lung, Breast)')
    ap.add_argument('--slides', help='csv with slide_id, organ[, age]; overrides --organ')
    ap.add_argument('--ckpt_dir', default='checkpoints')
    ap.add_argument('--weights', default='results/fusion_weights.csv')
    ap.add_argument('--align', action='store_true', help='mean-std align embeddings to GTEx, per organ')
    ap.add_argument('--ref_moments', default='data/gtex_moments.npz')
    ap.add_argument('--out', required=True)
    return ap.parse_args()


def load_folds(ckpt_dir, clock, device):
    return [load_clock(Path(ckpt_dir) / clock / f'fold{k}' / 'best_model.pt', device) for k in range(N_FOLDS)]


@torch.no_grad()
def score(models, x):
    return float(np.mean([m(x[None])[0].item() for m in models])) * AGE_STD + AGE_MEAN


def recalibration(y, p, k=10, n_splits=2000, seed=0):
    """held-out MAE after a linear fit on k labelled slides, and the age-only baseline
    (mean age of the same k slides)."""
    rng = np.random.default_rng(seed)
    cal, base = [], []
    for _ in range(n_splits):
        i = rng.permutation(len(y))
        tr, te = i[:k], i[k:]
        slope, icpt = np.polyfit(p[tr], y[tr], 1)
        cal.append(np.abs(slope * p[te] + icpt - y[te]).mean())
        base.append(np.abs(y[tr].mean() - y[te]).mean())
    return np.mean(cal), np.std(cal), np.mean(base)


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fdir = Path(args.features_dir)
    if args.slides:
        slides = pd.read_csv(args.slides)
    else:
        assert args.organ, 'give --organ or --slides'
        slides = pd.DataFrame({'slide_id': sorted(p.stem for p in fdir.glob('*.h5')), 'organ': args.organ})
    w = pd.read_csv(args.weights, index_col='organ')['w']
    pan = load_folds(args.ckpt_dir, 'pan', device)
    if args.align:
        ref = np.load(args.ref_moments)
        ref_mean, ref_sd = torch.from_numpy(ref['mean']).to(device), torch.from_numpy(ref['sd']).to(device)

    rows = []
    for organ, g in slides.groupby('organ', sort=False):
        organ_models = load_folds(args.ckpt_dir, organ, device) if organ in PER_ORGAN_CLOCKS else None
        w_o = float(w.get(organ, 0.0)) if organ_models else 0.0
        paths = [fdir / f'{s}.h5' for s in g.slide_id]
        if args.align:
            src_mean, src_sd = (t.to(device) for t in feature_moments(paths, device=device))
        print(f'{organ}: {len(g)} slides, w_o = {w_o:.2f}'
              + ('' if organ_models else ' (no per-organ clock, pan-organ only)'), flush=True)
        for (_, r), p in zip(g.iterrows(), paths):
            x = read_features(p).to(device)
            if args.align:
                x = align(x, src_mean, src_sd, ref_mean, ref_sd)
            y_pan = score(pan, x)
            y_org = score(organ_models, x) if organ_models else np.nan
            fused = y_pan if not organ_models else (1 - w_o) * y_pan + w_o * y_org
            rows.append({**r.to_dict(), 'pan': y_pan, 'per_organ': y_org, 'w_o': w_o, 'fused': fused})

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    print(f'wrote {args.out}')

    if 'age' in res and res.age.notna().any():
        for organ, g in res.dropna(subset=['age']).groupby('organ'):
            y, p = g.age.to_numpy(float), g.fused.to_numpy(float)
            msg = (f'{organ} (n={len(g)}): MAE {np.abs(p - y).mean():.2f}  r {np.corrcoef(p, y)[0, 1]:.2f}  '
                   f'mean bias {(p - y).mean():+.2f}')
            if len(g) > 12:
                cal, cal_sd, base = recalibration(y, p)
                msg += f'  | 10-slide recalibration MAE {cal:.2f} ± {cal_sd:.2f} (age-only {base:.2f})'
            print(msg)


if __name__ == '__main__':
    main()
