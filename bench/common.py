"""
PSF families and helpers shared by the benchmark scripts.

The paper uses three PSF sources, built by squarem_shear.psfs so the
benchmarks and the shear-bias runs use identical PSF models:

    superbit  SuperBIT PSFEx models (50 exposures, many positions each)
    hst       HST/ACS PSFs distributed with the GalSim COSMOS sample
    roman     GalSim Roman H158 design PSFs

Data locations are taken from the environment: SUPERBIT_PSFEX (a glob of
.psf files; default: the copies in a sibling ngmix-squarem-shear checkout)
and, optionally, COSMOS_DIR for the COSMOS sample.
"""
import os
import numpy as np
import ngmix
from ngmix.flags import EM_MAXITER
from squarem_shear.psfs import build_pool
from squarem_shear.guessers import AnyNGaussPSFGuesser

FAMILIES = {
    # name: (pixel scale in arcsec, stamp size)
    'superbit': (0.141, 48),
    'hst': (0.03, 64),
    'roman': (0.055, 64),
}


def psf_pool(family, npool, seed=31415):
    """
    the PSF pool for a family; entries are built lazily, so a chunk of a
    large pool only pays for the PSFs it uses
    """
    if family == 'superbit':
        files = os.environ.get(
            'SUPERBIT_PSFEX',
            '../ngmix-squarem-shear/psf_data/emp_psfs_best/psfex-output/*.psf')
        conf = {'source': 'superbit', 'files': files,
                'native_pixel_scale': FAMILIES['superbit'][0]}
    elif family == 'hst':
        conf = {'source': 'cosmos', 'sample': '25.2'}
        if 'COSMOS_DIR' in os.environ:
            conf['dir'] = os.environ['COSMOS_DIR']
    elif family == 'roman':
        conf = {'source': 'roman'}
    else:
        raise ValueError(family)
    conf.update(npool=npool, seed=seed)
    return build_pool({'psf': conf})


def chunk_range(n, chunk, nchunks):
    """indices of chunk `chunk` of `nchunks` over range(n)"""
    edges = np.linspace(0, n, nchunks + 1).astype(int)
    return range(edges[chunk], edges[chunk + 1])


def psf_obs(family, entry):
    scale, dims = FAMILIES[family]
    method = 'no_pixel' if entry['pixel_included'] else 'auto'
    im = entry['obj'].drawImage(
        nx=dims, ny=dims, scale=scale, method=method).array
    im /= im.sum()
    cen = (dims - 1) / 2
    jac = ngmix.DiagonalJacobian(row=cen, col=cen, scale=scale)
    wt = np.full(im.shape, 1.0 / (1.0e-5 * im.max())**2)
    return ngmix.Observation(im, weight=wt, jacobian=jac)


def guess(obs, seed, ngauss=5, ntry=20):
    """
    a valid starting mixture: the same sequence of GMixPSFGuesser draws a
    PSFRunner would make, keeping the first one EM can start from
    """
    rng = np.random.RandomState(seed)
    gsr = AnyNGaussPSFGuesser(rng=rng, ngauss=ngauss)
    for _ in range(ntry):
        g = gsr(obs=obs)
        r = ngmix.em.EMFitter(miniter=1, maxiter=1, tol=0.0).go(
            obs=obs, guess=g.copy())
        # one step from a valid start always reaches maxiter=1
        if r['flags'] & ~EM_MAXITER == 0:
            return g
    return None


def shape(gm):
    pars = gm.get_full_pars().reshape(-1, 6)
    p = pars[:, 0]
    irr, irc, icc = [(p * pars[:, k]).sum() / p.sum() for k in (3, 4, 5)]
    T = irr + icc
    return T, (icc - irr) / T, 2 * irc / T


def fp_residual(obs, gm):
    """
    scale-free fixed-point residual: how far one plain EM step moves the
    mixture's size and ellipticity, max(|dT|/T, |de1|, |de2|)
    """
    r = ngmix.em.EMFitter(miniter=1, maxiter=1, tol=0.0).go(
        obs=obs, guess=gm.copy())
    if not r.has_gmix():
        return np.nan
    a, b = shape(gm), shape(r.get_gmix())
    return max(abs(b[0] - a[0]) / a[0], abs(b[1] - a[1]), abs(b[2] - a[2]))


def model_quality(obs, gm):
    """
    how well the mixture represents the PSF image: the fractional L1
    residual, and the error in the adaptive-moment size and ellipticity of
    the model image relative to the PSF image (what a shape measurement sees)
    """
    model = gm.make_image(obs.image.shape, jacobian=obs.jacobian)
    resid = np.abs(model - obs.image).sum() / obs.image.sum()
    rng = np.random.RandomState(0)
    a = ngmix.admom.run_admom(obs, guess=4 * obs.jacobian.scale**2, rng=rng)
    mobs = ngmix.Observation(model, jacobian=obs.jacobian)
    b = ngmix.admom.run_admom(mobs, guess=4 * obs.jacobian.scale**2, rng=rng)
    if a['flags'] != 0 or b['flags'] != 0:
        return resid, np.nan, np.nan
    dT = (b['T'] - a['T']) / a['T']
    de = np.hypot(*(b['e'] - a['e']))
    return resid, dT, de
