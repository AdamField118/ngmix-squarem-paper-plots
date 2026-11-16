"""
Appendix C: do Moffat mixtures help?  Fit each PSF with

    ngmix EM, 5 Gaussians, plain and SQUAREM      (the production model)
    ngmix EM, 7 Gaussians, SQUAREM
    Moffat (Student-t) mixtures, 1-3 components, plain and SQUAREM
    Gaussian mixture, 7 components, SQUAREM, in the same implementation as
        the Moffat kernel (an implementation baseline for wall time)

all at tol 1e-6 and maxiter 5e4, and record iterations, wall time and the
fidelity of the model (adaptive-moment size and ellipticity errors and the
fractional L1 image residual).  Every method is warmed up (numba-compiled)
before anything is timed.

    python bench/run_moffat_bench.py --family superbit --npool 200 \
        --out data/bench/superbit_moffat_0000.npy
"""
import argparse
import time
import numpy as np
import ngmix

from common import psf_obs, psf_pool, guess, chunk_range
from squarem_shear import moffat_em

TOL = 1.0e-6
MAXITER = 50000

DTYPE = np.dtype([
    ('family', 'U10'), ('ipsf', 'i4'), ('method', 'U16'), ('ncomp', 'i4'),
    ('accelerate', '?'), ('flags', 'i4'), ('numiter', 'i8'), ('time', 'f8'),
    ('dT_admom', 'f8'), ('de_admom', 'f8'), ('resid', 'f8'),
    ('beta_min', 'f8'), ('beta_max', 'f8'),
])

METHODS = [  # (name, kind, ncomp, accelerate)
    ('em5', 'ngmix', 5, False),
    ('em5_sq', 'ngmix', 5, True),
    ('em7_sq', 'ngmix', 7, True),
    ('gauss7_sq', 'gauss', 7, True),
    ('moffat1', 'moffat', 1, False),
    ('moffat1_sq', 'moffat', 1, True),
    ('moffat2', 'moffat', 2, False),
    ('moffat2_sq', 'moffat', 2, True),
    ('moffat3', 'moffat', 3, False),
    ('moffat3_sq', 'moffat', 3, True),
]


def admom(obs, image):
    o = ngmix.Observation(image, jacobian=obs.jacobian)
    r = ngmix.admom.run_admom(o, guess=4 * obs.jacobian.scale**2,
                              rng=np.random.RandomState(0))
    return r if r['flags'] == 0 else None


def quality(obs, model, ref):
    model = model * obs.image.sum() / model.sum()
    resid = np.abs(model - obs.image).sum() / obs.image.sum()
    a = admom(obs, model)
    if ref is None or a is None:
        return np.nan, np.nan, resid
    return (a['T'] - ref['T']) / ref['T'], np.hypot(*(a['e'] - ref['e'])), \
        resid


def run_method(obs, kind, ncomp, acc, seed):
    if kind == 'ngmix':
        g0 = guess(obs, seed=seed, ngauss=ncomp)
        t0 = time.perf_counter()
        r = ngmix.em.EMFitter(tol=TOL, maxiter=MAXITER,
                              accelerate=acc).go(obs=obs, guess=g0)
        dt = time.perf_counter() - t0
        if not r.has_gmix():
            return None
        m = r.get_gmix().make_image(obs.image.shape, jacobian=obs.jacobian)
        return dict(model=m, numiter=r['numiter'], time=dt,
                    flags=int(r['flags']), beta=(np.nan, np.nan))
    r = moffat_em.fit(obs, ncomp, gaussian=(kind == 'gauss'), tol=TOL,
                      maxiter=MAXITER, accelerate=acc, seed=seed)
    b = moffat_em.beta(r['params'], ncomp)
    return dict(model=r['model'], numiter=r['numiter'], time=r['time'],
                flags=r['flags'],
                beta=(b.min(), b.max()) if kind == 'moffat' else (np.nan,) * 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True)
    p.add_argument('--npool', type=int, default=200)
    p.add_argument('--chunk', type=int, default=0)
    p.add_argument('--nchunks', type=int, default=1)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    pool = psf_pool(args.family, args.npool, seed=16180)
    obs = psf_obs(args.family, pool[0])
    for _, kind, nc, acc in METHODS:          # compile before timing
        run_method(obs, kind, nc, acc, seed=0)

    rows = []
    t0 = time.time()
    idx = chunk_range(len(pool), args.chunk, args.nchunks)
    for n, i in enumerate(idx):
        obs = psf_obs(args.family, pool[i])
        ref = admom(obs, obs.image)
        for name, kind, nc, acc in METHODS:
            r = run_method(obs, kind, nc, acc, seed=3000 + i)
            if r is None:
                continue
            q = quality(obs, r['model'], ref)
            rows.append((args.family, i, name, nc, acc, r['flags'],
                         r['numiter'], r['time'], *q, *r['beta']))
        if (n + 1) % 10 == 0:
            print('%s %d/%d  %.0f s' % (args.family, n + 1, len(idx),
                                        time.time() - t0), flush=True)
    np.save(args.out, np.array(rows, dtype=DTYPE))


if __name__ == '__main__':
    main()
