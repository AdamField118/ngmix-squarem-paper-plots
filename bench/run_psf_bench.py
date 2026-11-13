"""
Fit every PSF in a family with plain EM and SQUAREM, at several tolerances,
from the same starting mixture, and record cost and quality.

    # one family, all PSFs, in one process
    python bench/run_psf_bench.py --family superbit --npool 1000 \
        --out data/bench/superbit_0000.npy

    # the same split into 20 jobs (see scripts/slurm_bench.sh), then
    python bench/merge.py data/bench/superbit_*.npy --out data/psf_bench.npy

Every row is one (PSF, ngauss, tol, runner) fit.
"""
import argparse
import time
import numpy as np
import ngmix
from ngmix.flags import EM_MAXITER

from common import (psf_pool, psf_obs, guess, shape, fp_residual,
                    model_quality, chunk_range)

TOLS = (1.0e-5, 1.0e-6, 1.0e-8)

DTYPE = np.dtype([
    ('family', 'U10'), ('ipsf', 'i4'), ('ngauss', 'i4'), ('tol', 'f8'),
    ('accelerate', '?'), ('flags', 'i4'), ('maxiter_hit', '?'),
    ('numiter', 'i8'), ('time', 'f8'), ('fp', 'f8'), ('T', 'f8'),
    ('e1', 'f8'), ('e2', 'f8'), ('resid', 'f8'), ('dT_admom', 'f8'),
    ('de_admom', 'f8'), ('loglike', 'f8'),
])


def loglike(obs, gm):
    """
    Poisson-like log likelihood of the image under the mixture, sum d log m,
    the quantity EM increases; used to compare the optima the runners reach
    """
    m = gm.make_image(obs.image.shape, jacobian=obs.jacobian)
    m = m / m.sum()
    d = obs.image
    good = (d > 0) & (m > 0)
    return float((d[good] * np.log(m[good])).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True)
    p.add_argument('--npool', type=int, default=1000)
    p.add_argument('--chunk', type=int, default=0)
    p.add_argument('--nchunks', type=int, default=1)
    p.add_argument('--ngauss', type=int, nargs='+', default=[5])
    p.add_argument('--tols', type=float, nargs='+', default=list(TOLS))
    p.add_argument('--maxiter', type=int, default=50000)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    pool = psf_pool(args.family, args.npool)
    rows = []
    t0 = time.time()
    idx = chunk_range(len(pool), args.chunk, args.nchunks)
    for n, i in enumerate(idx):
        obs = psf_obs(args.family, pool[i])
        for ng in args.ngauss:
            g0 = guess(obs, seed=1000 + i, ngauss=ng)
            if g0 is None:
                continue
            for tol in args.tols:
                for acc in (False, True):
                    f = ngmix.em.EMFitter(tol=tol, maxiter=args.maxiter,
                                          accelerate=acc)
                    t = time.perf_counter()
                    r = f.go(obs=obs, guess=g0.copy())
                    dt = time.perf_counter() - t
                    ok = r.has_gmix()
                    gm = r.get_gmix() if ok else None
                    nan3 = (np.nan,) * 3
                    rows.append((
                        args.family, i, ng, tol, acc, r['flags'],
                        bool(r['flags'] & EM_MAXITER),
                        r.get('numiter', -1), dt,
                        fp_residual(obs, gm) if ok else np.nan,
                        *(shape(gm) if ok else nan3),
                        *(model_quality(obs, gm) if ok else nan3),
                        loglike(obs, gm) if ok else np.nan,
                    ))
        if (n + 1) % 25 == 0:
            print('%s %d/%d  %.0f s' % (args.family, n + 1, len(idx),
                                        time.time() - t0), flush=True)

    np.save(args.out, np.array(rows, dtype=DTYPE))


if __name__ == '__main__':
    main()
