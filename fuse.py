"""Cross-fitted organ-wise fusion of the pan-organ and per-organ clocks.

    y_hat = (1 - w_o) * y_pan + w_o * y_organ,     w_o in [0, 1] (grid step 0.01)

For validation fold k, w_o is chosen to minimize the MAE of the out-of-fold predictions of the
other four folds and then applied to fold k, so no slide contributes to the weight applied to it.
Organs without a per-organ clock keep w_o = 0 (pan-organ prediction only).

    python fuse.py --ckpt_dir checkpoints --out results

Reads   <ckpt_dir>/pan/fold<k>/val_predictions.csv and <ckpt_dir>/<organ>/fold<k>/val_predictions.csv
Writes  <out>/oof_predictions.csv  one row per slide: pan, per_organ, fused (all out-of-fold)
        <out>/fusion_weights.csv   per organ: w_o of each fold and their mean (used by predict.py)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hera.config import PER_ORGAN_CLOCKS

GRID = np.linspace(0, 1, 101)
N_FOLDS = 5


def best_weight(y, pan, organ):
    return GRID[int(np.argmin([np.abs(y - ((1 - w) * pan + w * organ)).mean() for w in GRID]))]


def metrics(pred, df):
    y = df.age.to_numpy()
    e = np.asarray(pred) - y
    ae = np.abs(e)
    return dict(mae=ae.mean(),
                mae_tissue_avg=pd.Series(ae).groupby(df.tissue.values).mean().mean(),
                r=np.corrcoef(pred, y)[0, 1],
                r2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt_dir', default='checkpoints')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    ck, out = Path(args.ckpt_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pan = pd.concat([pd.read_csv(ck / 'pan' / f'fold{k}' / 'val_predictions.csv') for k in range(N_FOLDS)],
                    ignore_index=True).rename(columns={'pred': 'pan'})
    po = pd.concat([pd.read_csv(ck / o / f'fold{k}' / 'val_predictions.csv')[['slide_id', 'pred']]
                    for o in PER_ORGAN_CLOCKS for k in range(N_FOLDS)],
                   ignore_index=True).rename(columns={'pred': 'per_organ'})
    assert pan.slide_id.is_unique and po.slide_id.is_unique
    m = pan.merge(po, on='slide_id', how='left')
    has = m.per_organ.notna().to_numpy()
    print(f'{len(m)} slides, {has.sum()} with a per-organ clock')

    m['fused'] = m.pan
    weights = []
    for k in range(N_FOLDS):
        test = (m.fold == k).to_numpy()
        for organ, d in m[~test & has].groupby('organ'):
            w = best_weight(d.age.to_numpy(), d.pan.to_numpy(), d.per_organ.to_numpy())
            weights.append(dict(organ=organ, fold=k, w=w))
            sel = test & has & (m.organ == organ).to_numpy()
            m.loc[sel, 'fused'] = (1 - w) * m.pan[sel] + w * m.per_organ[sel]

    tab = pd.DataFrame({'pan-organ': metrics(m.pan, m), 'fused': metrics(m.fused, m)}).T
    print('\nout-of-fold, pooled over 5 folds')
    print(tab.round(3).to_string())

    w = pd.DataFrame(weights).pivot(index='organ', columns='fold', values='w')
    w.columns = [f'w_fold{k}' for k in w.columns]
    w['w'] = w.mean(axis=1)
    per_organ = m[has].groupby('organ').apply(lambda d: pd.Series({
        'n': len(d), 'mae_pan': (d.pan - d.age).abs().mean(),
        'mae_per_organ': (d.per_organ - d.age).abs().mean(),
        'mae_fused': (d.fused - d.age).abs().mean()}), include_groups=False)
    w = per_organ.join(w).sort_values('w', ascending=False)
    print('\nper organ')
    print(w.round(3).to_string())

    m.to_csv(out / 'oof_predictions.csv', index=False)
    w.to_csv(out / 'fusion_weights.csv')
    print(f'\nwrote {out / "oof_predictions.csv"} and {out / "fusion_weights.csv"}')


if __name__ == '__main__':
    main()
