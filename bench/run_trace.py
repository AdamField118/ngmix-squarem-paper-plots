"""
Convergence traces: for each of `--npsf` PSFs, run both runners with tol=0
to a grid of iteration caps and record, at each stopping point, em_run's own
convergence statistic |dlogL/logL| between the last two plain steps and the
scale-free fixed-point residual.

    python bench/run_trace.py --family superbit --npsf 100 \
        --out data/trace/superbit.npz

Output arrays have shape (npsf, ngrid).
"""
import argparse
import time
import numpy as np
import ngmix

from common import psf_pool, psf_obs, guess, fp_residual


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True)
    p.add_argument('--npsf', type=int, default=100)
    p.add_argument('--maxiter', type=int, default=50000)
    p.add_argument('--npoints', type=int, default=40)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    grid = np.unique(np.geomspace(2, args.maxiter, args.npoints).astype(int))
    pool = psf_pool(args.family, args.npsf, seed=27182)
    out = {'grid': grid}
    for runner in ('plain', 'squarem'):
        for k in ('fdiff', 'fp', 'numiter'):
            out['%s/%s' % (runner, k)] = np.full((args.npsf, grid.size),
                                                 np.nan)
    t0 = time.time()
    for i in range(args.npsf):
        obs = psf_obs(args.family, pool[i])
        g0 = guess(obs, seed=2000 + i)
        if g0 is None:
            continue
        for runner in ('plain', 'squarem'):
            for j, k in enumerate(grid):
                r = ngmix.em.EMFitter(
                    miniter=1, maxiter=int(k), tol=0.0,
                    accelerate=runner == 'squarem',
                ).go(obs=obs, guess=g0.copy())
                if not r.has_gmix():
                    continue
                out[runner + '/fdiff'][i, j] = r.get('fdiff', np.nan)
                out[runner + '/fp'][i, j] = fp_residual(obs, r.get_gmix())
                out[runner + '/numiter'][i, j] = r.get('numiter', -1)
        print('%s %d/%d  %.0f s' % (args.family, i + 1, args.npsf,
                                    time.time() - t0), flush=True)
    np.savez(args.out, **out)


if __name__ == '__main__':
    main()
