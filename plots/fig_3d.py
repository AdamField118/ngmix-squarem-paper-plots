"""
Appendix C: a PSF and its Gaussian-mixture fits as 3D surfaces.

For one PSF from each requested source: the PSF image, the em5 fit from plain
EM and from SQUAREM, and SQUAREM fits with 7 and 9 components, all from the
same starting mixture at the SuperBIT settings (tol 1e-6, maxiter 5e4).
Top row: log10 of the intensity relative to the PSF peak.  Bottom row: the
residual, fit minus PSF, in units of the PSF peak, on a common scale.

    python plots/fig_3d.py figures --families superbit roman hst --ipsf 0

Writes figures/psf3d_<family>.pdf.  Needs the PSF data (see bench/common.py).
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import ngmix

import style

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'bench'))
from common import psf_pool, psf_obs, guess  # noqa: E402

FITS = [  # (label, ngauss, accelerate)
    ('EM, 5 Gaussians', 5, False),
    ('SQUAREM, 5', 5, True),
    ('SQUAREM, 7', 7, True),
    ('SQUAREM, 9', 9, True),
]
LOG_FLOOR = -4.0     # decades below the peak shown in the top row
HALF = 12            # pixels either side of the centre that are shown


def fit(obs, ngauss, accelerate, seed):
    g0 = guess(obs, seed=seed, ngauss=ngauss)
    r = ngmix.em.EMFitter(tol=1.0e-6, maxiter=50000,
                          accelerate=accelerate).go(obs=obs, guess=g0)
    gm = r.get_gmix()
    model = gm.make_image(obs.image.shape, jacobian=obs.jacobian)
    return model, r['numiter']


def crop(a):
    c = a.shape[0] // 2
    return a[c - HALF:c + HALF + 1, c - HALF:c + HALF + 1]


def surface(ax, X, Y, Z, cmap, zlim, norm=None):
    ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm, rstride=1, cstride=1,
                    linewidth=0, antialiased=True)
    ax.set_zlim(*zlim)
    ax.view_init(elev=28, azim=-60)
    ax.set_box_aspect((1, 1, 0.7))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.set_tick_params(labelsize=5, pad=0)
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    if zlim[0] == LOG_FLOOR:
        ax.set_zticks([-4, -2, 0])


def figure(family, ipsf, outdir):
    pool = psf_pool(family, ipsf + 1)
    obs = psf_obs(family, pool[ipsf])
    im = obs.image / obs.image.sum()
    peak = im.max()

    models = []
    for label, ng, acc in FITS:
        m, n = fit(obs, ng, acc, seed=1000 + ipsf)
        m = m / m.sum()
        l1 = np.abs(m - im).sum()
        models.append((label, m, n, l1))

    y, x = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
    ncol = 1 + len(models)
    fig = plt.figure(figsize=(style.PAGE_W, 3.1))
    rmax = max(np.abs(crop(m - im)).max() for _, m, _, _ in models) / peak

    def logim(a):
        return np.log10(np.clip(crop(a) / peak, 10**LOG_FLOOR, None))

    ax = fig.add_subplot(2, ncol, 1, projection='3d')
    surface(ax, x, y, logim(im), 'Blues', (LOG_FLOOR, 0))
    ax.set_title('%s PSF' % style.FAMILY_LABELS[family], fontsize=7)
    ax.set_zlabel(r'$\log_{10} I/I_{\rm peak}$', fontsize=6, labelpad=-2)
    for j, (label, m, n, l1) in enumerate(models):
        ax = fig.add_subplot(2, ncol, j + 2, projection='3d')
        surface(ax, x, y, logim(m), 'Blues', (LOG_FLOOR, 0))
        ax.set_title('%s\n%d iterations, $L_1$ resid. %.3f' % (label, n, l1),
                     fontsize=6.5)
        ax = fig.add_subplot(2, ncol, ncol + j + 2, projection='3d')
        surface(ax, x, y, crop(m - im) / peak, 'RdBu_r', (-rmax, rmax),
                norm=plt.Normalize(-rmax, rmax))
        ax.set_zticks([-rmax, 0, rmax])
        ax.set_zticklabels(['%.2f' % -rmax, '0', '%.2f' % rmax])
        if j == 0:
            ax.set_zlabel(r'(fit $-$ PSF)$/I_{\rm peak}$', fontsize=6,
                          labelpad=-2)
    ax = fig.add_subplot(2, ncol, ncol + 1)
    ax.axis('off')
    ax.text(0.5, 0.5, 'residuals\n(common scale,\n$\\pm$%.3f of peak)' % rmax,
            ha='center', va='center', fontsize=7, color=style.REF)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.02,
                        wspace=0.02, hspace=0.12)
    out = os.path.join(outdir, 'psf3d_%s.pdf' % family)
    fig.savefig(out)
    print(out, [(lab, n, round(l1, 4)) for lab, _, n, l1 in models])


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
            # e.g. the COSMOS sample is not installed: skip that source
            print('skipping %s: %s' % (fam, e))


if __name__ == '__main__':
    main()
