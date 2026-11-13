"""
Iterations to convergence for each PSF family at the SuperBIT tolerance, as
empirical CDFs, and the per-PSF speedup at three tolerances.

    python plots/fig_iterations.py data/psf_bench.npy figures/
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import style

style.setup()
d = np.load(sys.argv[1])
outdir = sys.argv[2]
d = d[d['ngauss'] == 5]
fams = [f for f in style.FAMILY_LABELS if f in set(d['family'])]


def ecdf(ax, x, **kw):
    x = np.sort(x)
    ax.step(x, np.arange(1, x.size + 1) / x.size, where='post', **kw)


fig, axes = plt.subplots(1, len(fams), figsize=(style.PAGE_W, 1.9),
                         sharey=True)
for ax, fam in zip(axes, fams):
    for acc, sty in ((False, style.PLAIN), (True, style.SQUAREM)):
        x = d[(d['family'] == fam) & (d['tol'] == style.TOL_SUPERBIT)
              & (d['accelerate'] == acc)]
        ecdf(ax, x['numiter'], color=sty['color'], ls=sty['ls'],
             label=sty['label'])
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(style.FAMILY_LABELS[fam])
    ax.set_xlabel(r'iterations to tol $=10^{-6}$')
axes[0].set_ylabel('fraction of PSFs')
axes[-1].legend(loc='lower right')
fig.savefig(outdir + '/iterations_cdf.pdf')

# speedup per PSF: ratio of iteration counts, same start, same tol
tols = (1.0e-5, 1.0e-6, 1.0e-8)
fig, ax = plt.subplots(figsize=(style.COL_W, 2.3))
width = 0.8 / len(tols)
for k, tol in enumerate(tols):
    for j, fam in enumerate(fams):
        sel = (d['family'] == fam) & (d['tol'] == tol)
        a = d[sel & ~d['accelerate']]
        b = d[sel & d['accelerate']]
        r = a['numiter'] / b['numiter']
        x = j + (k - 1) * width
        q10, q50, q90 = np.percentile(r, [10, 50, 90])
        ax.plot([x, x], [q10, q90], color=style.SQUAREM['color'], lw=1.0,
                alpha=0.4 + 0.3 * k)
        ax.plot(x, q50, marker='os^'[k], ms=4, color=style.SQUAREM['color'],
                ls='none', label=r'tol $=10^{%d}$' % np.log10(tol)
                if j == 0 else None)
ax.axhline(1.0, color=style.REF, ls=':', lw=0.8)
ax.set_yscale('log')
ax.set_xticks(range(len(fams)))
ax.set_xticklabels([style.FAMILY_LABELS[f].split(' (')[0] for f in fams],
                   rotation=0)
ax.set_ylabel('EM / SQUAREM iterations')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3,
          handletextpad=0.2, columnspacing=0.8)
fig.savefig(outdir + '/speedup.pdf')
