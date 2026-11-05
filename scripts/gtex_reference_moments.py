"""GTEx reference moments for predict.py --align (already provided as data/gtex_moments.npz).

300 randomly chosen GTEx slides, up to 2,000 patches each.

    python scripts/gtex_reference_moments.py --features_dir <gtex features_uni_v2> --out data/gtex_moments.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hera.align import feature_moments  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features_dir', required=True)
    ap.add_argument('--n_slides', type=int, default=300)
    ap.add_argument('--out', default='data/gtex_moments.npz')
    args = ap.parse_args()
    paths = sorted(Path(args.features_dir).glob('*.h5'))
    pick = np.random.default_rng(0).choice(len(paths), args.n_slides, replace=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean, sd = feature_moments([paths[i] for i in pick], device=device)
    np.savez(args.out, mean=mean.numpy(), sd=sd.numpy())
    print(f'wrote {args.out} from {args.n_slides} of {len(paths)} slides')


if __name__ == '__main__':
    main()
