"""Attach donor age to the public slide list.

data/gtex_splits.csv (slide_id, subject_id, tissue, organ, fold) is public. Exact donor age is
controlled-access GTEx data (dbGaP phs000424, file GTEx_Subject_Phenotypes.GRU.txt), so it is
joined here and never stored in the repository.

    python scripts/prepare_metadata.py --phenotypes /path/to/GTEx_Subject_Phenotypes.GRU.txt
    -> data/metadata.csv (gtex_splits.csv + age)
"""
import argparse

import pandas as pd


def read_dbgap_table(path, first_col='dbGaP_Subject_ID'):
    """dbGaP tables start with a variable-length comment block; find the header row."""
    with open(path) as fh:
        for i, line in enumerate(fh):
            if line.split('\t')[0].strip() == first_col:
                return pd.read_csv(path, sep='\t', skiprows=i, low_memory=False)
    raise ValueError(f'no header row starting with {first_col} in {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phenotypes', required=True, help='GTEx_Subject_Phenotypes.GRU.txt (dbGaP)')
    ap.add_argument('--splits', default='data/gtex_splits.csv')
    ap.add_argument('--out', default='data/metadata.csv')
    args = ap.parse_args()

    splits = pd.read_csv(args.splits)
    subj = read_dbgap_table(args.phenotypes)[['SUBJID', 'AGE']]
    meta = splits.merge(subj.rename(columns={'SUBJID': 'subject_id', 'AGE': 'age'}),
                        on='subject_id', how='left', validate='m:1')
    missing = meta.age.isna().sum()
    assert missing == 0, f'{missing} slides without age - wrong phenotype file?'
    meta.to_csv(args.out, index=False)
    print(f'wrote {args.out}: {len(meta)} slides, {meta.subject_id.nunique()} donors, '
          f'age {meta.age.min():.0f}-{meta.age.max():.0f}')


if __name__ == '__main__':
    main()
