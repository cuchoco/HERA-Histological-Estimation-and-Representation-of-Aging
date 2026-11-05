"""Train one histological age clock on one cross-validation fold.

    python train.py --clock pan    --fold 0     # pan-organ clock: all slides of all 40 tissue types
    python train.py --clock Brain  --fold 0     # per-organ clock: one organ's slides only
    python train.py --clock organs --fold 0     # all 24 per-organ clocks, one after another

Both clocks share architecture and schedule: AdamW (lr 1e-4, weight decay 1e-5), cosine decay,
batch of 4 slides, MSE on standardized age, 20 epochs. The epoch with the lowest validation MAE is
kept; for the pan-organ clock that MAE is averaged over the 40 tissue types, for a per-organ clock
it is pooled over the organ's slides. The defaults of --seed and --grad_clip are the settings the
paper's checkpoints were trained with (pan: seed 42, clip 5; per-organ: seed 0, no clipping).

Output: <out>/<clock>/fold<k>/best_model.pt, val_predictions.csv, history.csv
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hera.config import AGE_MEAN, AGE_STD, PER_ORGAN_CLOCKS
from hera.data import SlideDataset, collate
from hera.model import AgeClock


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clock', required=True, help="'pan', 'organs' (all 24) or an organ name, e.g. Brain")
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--metadata', default='data/metadata.csv')
    ap.add_argument('--features_dir', required=True, help='directory of TRIDENT <slide_id>.h5 files')
    ap.add_argument('--out', default='checkpoints')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=None, help='default: 42 (pan) / 0 (per-organ)')
    ap.add_argument('--grad_clip', type=float, default=None, help='default: 5 (pan) / off (per-organ)')
    return ap.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _, mask in loader:
        y, _ = model(x.to(device), mask.to(device))
        out.append(y.float().cpu().numpy())
    return np.concatenate(out) * AGE_STD + AGE_MEAN


def train_clock(args, clock, meta, device):
    is_pan = clock == 'pan'
    seed = args.seed if args.seed is not None else (42 if is_pan else 0)
    grad_clip = args.grad_clip if args.grad_clip is not None else (5.0 if is_pan else 0.0)
    out = Path(args.out) / clock / f'fold{args.fold}'
    if (out / 'history.csv').exists():
        print(f'[skip] {out} already trained')
        return
    out.mkdir(parents=True, exist_ok=True)

    d = meta if is_pan else meta[meta.organ == clock]
    tr_df, va_df = d[d.fold != args.fold], d[d.fold == args.fold].reset_index(drop=True)
    print(f'=== {clock} fold {args.fold}: train {len(tr_df)} | val {len(va_df)} slides '
          f'| seed {seed} | grad_clip {grad_clip or "off"}', flush=True)

    set_seed(seed)
    model = AgeClock().to(device)
    kw = dict(batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
    tr = DataLoader(SlideDataset(tr_df, args.features_dir), shuffle=True, **kw)
    va = DataLoader(SlideDataset(va_df, args.features_dir), shuffle=False, **kw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(tr) * args.epochs)

    y_va = va_df.age.to_numpy()
    best, hist = float('inf'), []
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        run = 0.0
        for x, y, mask in tr:
            pred, _ = model(x.to(device), mask.to(device))
            loss = F.mse_loss(pred, y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            run += loss.item()

        p = predict(model, va, device)
        ae = np.abs(p - y_va)
        mae, macro = ae.mean(), pd.Series(ae).groupby(va_df.tissue.values).mean().mean()
        score = macro if is_pan else mae      # model-selection criterion
        hist.append(dict(epoch=ep + 1, train_loss=run / len(tr), val_mae=mae,
                         val_mae_tissue_avg=macro, minutes=(time.time() - t0) / 60))
        flag = ''
        if score < best:
            best, flag = score, '  *best'
            torch.save({'model_state_dict': model.state_dict(), 'clock': clock, 'fold': args.fold,
                        'epoch': ep + 1, 'val_score': score}, out / 'best_model.pt')
            va_df.assign(pred=p).to_csv(out / 'val_predictions.csv', index=False)
        print(f'  ep{ep + 1:02d}  train {run / len(tr):.4f}  val MAE {mae:.3f}  '
              f'tissue-avg {macro:.3f}{flag}', flush=True)
    pd.DataFrame(hist).to_csv(out / 'history.csv', index=False)
    print(f'<<< {clock} fold {args.fold}: best {best:.3f}', flush=True)


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta = pd.read_csv(args.metadata)
    clocks = PER_ORGAN_CLOCKS if args.clock == 'organs' else [args.clock]
    for c in clocks:
        if c != 'pan' and c not in PER_ORGAN_CLOCKS:
            raise SystemExit(f'unknown clock {c!r}; choose pan, organs or one of {PER_ORGAN_CLOCKS}')
        train_clock(args, c, meta, device)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
