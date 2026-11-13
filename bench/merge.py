"""
Concatenate benchmark chunks.

    python bench/merge.py data/bench/*_ng5_*.npy --out data/psf_bench.npy
"""
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    d = np.concatenate([np.load(f) for f in sorted(args.files)])
    np.save(args.out, d)
    for fam in np.unique(d['family']):
        print(fam, len(np.unique(d['ipsf'][d['family'] == fam])), 'psfs')


if __name__ == '__main__':
    main()
