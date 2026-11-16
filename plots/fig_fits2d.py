"""
Appendix D: a PSF and three models of it, in the style of Figure C1 of
Lensing in the Blue III.

Columns: the PSF; em5 fit by plain EM at the SuperBIT settings (the
production model); a 7-Gaussian SQUAREM fit; a 3-Moffat SQUAREM fit.  Top:
intensity on a shared logarithmic scale.  Bottom: model minus PSF, in units
of the PSF peak, on a shared symmetric scale.  All fits at tol 1e-6.

    python plots/fig_fits2d.py figures --families superbit hst roman

Writes figures/fits2d_<family>.pdf; sources whose data are missing are
skipped.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import ngmix

import style

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'bench'))
from common import psf_pool, psf_obs, guess  # noqa: E402
from squarem_shear import moffat_em  # noqa: E402

NMOFFAT = 3
DECADES = 4.5      # dynamic range shown on the log scale


def ngmix_fit(obs, ngauss, accelerate, seed):
    g0 = guess(obs, seed=seed, ngauss=ngauss)
    r = ngmix.em.EMFitter(tol=1.0e-6, maxiter=50000,
                          accelerate=accelerate).go(obs=obs, guess=g0)
    m = r.get_gmix().make_image(obs.image.shape, jacobian=obs.jacobian)
    return m, r['numiter']


def figure(family, ipsf, outdir):
    pool = psf_pool(family, ipsf + 1)
    obs = psf_obs(family, pool[ipsf])
    im = obs.image / obs.image.sum()
    seed = 1000 + ipsf

    panels = [('PSF (%s)' % style.FAMILY_LABELS[family], im, None)]
    m, n = ngmix_fit(obs, 5, False, seed)
    panels.append(('em5, plain EM', m, n))
    m, n = ngmix_fit(obs, 7, True, seed)
    panels.append(('7 Gaussians, SQUAREM', m, n))
    r = moffat_em.fit(obs, NMOFFAT, seed=seed)
    panels.append(('%d Moffats, SQUAREM' % NMOFFAT, r['model'],
                   r['numiter']))
    panels = [(t, a / a.sum(), n) for t, a, n in panels]

    peak = im.max()
    lognorm = LogNorm(vmin=peak * 10**-DECADES, vmax=peak)
    rmax = max(np.abs(a - im).max() for _, a, n in panels[1:]) / peak
    rnorm = Normalize(vmin=-rmax, vmax=rmax)
    half = im.shape[0] * obs.jacobian.scale / 2
    ext = (-half, half, -half, half)

    fig, axes = plt.subplots(2, len(panels), figsize=(style.PAGE_W, 3.7),
                             gridspec_kw={'hspace': 0.12, 'wspace': 0.06})
    for j, (title, a, n) in enumerate(panels):
        ax = axes[0, j]
        hi = ax.imshow(np.clip(a, peak * 10**-DECADES, None), norm=lognorm,
                       cmap='Blues', origin='lower', extent=ext)
        if n is None:
            ax.set_title(title, fontsize=6.5)
        else:
            l1 = np.abs(a - im).sum()
            ax.set_title('%s\n%d iter., $L_1$ = %.3f' % (title, n, l1),
                         fontsize=6.5)
        ax = axes[1, j]
        if j == 0:
            ax.axis('off')
            continue
        hr = ax.imshow((a - im) / peak, norm=rnorm, cmap='RdBu_r',
                       origin='lower', extent=ext)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel('intensity', fontsize=7)
    axes[1, 1].set_ylabel('model $-$ PSF', fontsize=7)
    c = fig.colorbar(hi, ax=axes[0, :], fraction=0.02, pad=0.01)
    c.set_label(r'$I / \sum I$', fontsize=7)
    c.ax.tick_params(labelsize=6)
    c = fig.colorbar(hr, ax=axes[1, :], fraction=0.02, pad=0.01)
    c.set_label(r'$\Delta I / I_{\rm peak}$', fontsize=7)
    c.ax.tick_params(labelsize=6)
    # a scale bar instead of axes: 0.5 arcsec
    axes[0, 0].plot([-half * 0.85, -half * 0.85 + 0.5], [-half * 0.85] * 2,
                    color=style.REF, lw=1.5)
    axes[0, 0].text(-half * 0.85 + 0.25, -half * 0.78,
                    '0.5$^{\\prime\\prime}$', ha='center', va='bottom',
                    fontsize=6, color=style.REF)
    out = os.path.join(outdir, 'fits2d_%s.pdf' % family)
    fig.savefig(out)
    print(out, [(t, n) for t, _, n in panels[1:]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('outdir')
    p.add_argument('--families', nargs='+', default=['superbit'])
    p.add_argument('--ipsf', type=int, default=0)
    args = p.parse_args()
    style.setup()
    for fam in args.families:
        try:
            figure(fam, args.ipsf, args.outdir)
        except (IOError, OSError) as e:
            print('skipping %s: %s' % (fam, e))


if __name__ == '__main__':
    main()
