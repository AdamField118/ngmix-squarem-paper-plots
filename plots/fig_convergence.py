"""
Figure 1 (background): em_run's convergence statistic |dlogL/logL| against
iteration for plain EM on SuperBIT PSFs: median over PSFs and the 16-84th
percentile band, with the ngmix default and SuperBIT tolerances.

Figure 3 (results): both runners, both convergence measures, each family.

    python plots/fig_convergence.py data/trace figures
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import style

style.setup()
tracedir, outdir = sys.argv[1], sys.argv[2]
traces = {f: np.load(os.path.join(tracedir, f + '.npz'))
          for f in style.FAMILIES
          if os.path.exists(os.path.join(tracedir, f + '.npz'))}


def band(ax, grid, x, sty, label=None):
    """median and 16-84th percentile band over PSFs, ignoring zeros (a
    statistic of exactly zero cannot be drawn on a log axis)"""
    x = np.where(x > 0, x, np.nan)
    with np.errstate(all='ignore'):
        lo, med, hi = np.nanpercentile(x, [16, 50, 84], axis=0)
    ax.fill_between(grid, lo, hi, color=sty['color'], alpha=0.18, lw=0)
    ax.plot(grid, med, color=sty['color'], ls=sty['ls'], label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')


def tol_lines(ax, grid, label=True):
    ax.axhline(style.TOL_SUPERBIT, color=style.REF, ls=':', lw=1.0)
    ax.axhline(style.TOL_DEFAULT, color=style.REF, ls='-.', lw=0.8)
    if label:
        ax.text(grid[-1], style.TOL_SUPERBIT * 1.5,
                r'SuperBIT tol $=10^{-6}$', color=style.REF, fontsize=6.5,
                ha='right')
        ax.text(grid[-1], style.TOL_DEFAULT * 1.5,
                r'ngmix default tol $=10^{-5}$', color=style.REF,
                fontsize=6.5, ha='right')


# ---- figure 1: plain EM only, SuperBIT PSFs
fam = 'superbit' if 'superbit' in traces else list(traces)[0]
d = traces[fam]
grid = d['grid']
npsf = np.isfinite(d['plain/fdiff']).any(axis=1).sum()
fig, ax = plt.subplots(figsize=(style.COL_W, 2.4))
band(ax, grid, d['plain/fdiff'], style.PLAIN)
tol_lines(ax, grid)
ax.set_xlabel('EM iteration')
ax.set_ylabel(r'$|\Delta \log L / \log L|$')
ax.set_title('em5 fits to %d %s PSFs' % (npsf, style.FAMILY_LABELS[fam]))
fig.savefig(outdir + '/convergence_em.pdf')

# ---- figure 3: both runners, both statistics, every family
fams = list(traces)
fig, axes = plt.subplots(2, len(fams), figsize=(style.PAGE_W, 3.6),
                         sharex=True, sharey='row', squeeze=False)
for j, fam in enumerate(fams):
    d = traces[fam]
    for runner, sty in (('plain', style.PLAIN), ('squarem', style.SQUAREM)):
        band(axes[0, j], d['grid'], d[runner + '/fdiff'], sty,
             label=sty['label'])
        band(axes[1, j], d['grid'], d[runner + '/fp'], sty)
    tol_lines(axes[0, j], d['grid'], label=False)
    axes[0, j].set_title(style.FAMILY_LABELS[fam])
    axes[1, j].set_xlabel('EM iterations')
axes[0, 0].set_ylabel(r'$|\Delta \log L / \log L|$')
axes[1, 0].set_ylabel('fixed-point residual')
axes[0, 0].legend(loc='lower left')
fig.savefig(outdir + '/convergence_compare.pdf')
