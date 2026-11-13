"""
Appendix: does a richer mixture help, now that convergence is cheap?
PSF model fidelity (adaptive-moment size error and image residual) and cost
against the number of gaussians, with SQUAREM at the SuperBIT tolerance.

    python plots/fig_ngauss.py data/psf_bench_ngauss.npy figures/
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import style

style.setup()
d = np.load(sys.argv[1])
outdir = sys.argv[2]
d = d[d['tol'] == style.TOL_SUPERBIT]
fams = [f for f in style.FAMILY_LABELS if f in set(d['family'])]
ngs = sorted(set(d['ngauss']))
marks = dict(zip(fams, 'osD^'))

panels = [('dT_admom', r'$|\delta T/T|$ (adaptive moments)', True),
          ('resid', 'fractional image residual', True),
          ('numiter', 'iterations', True)]
fig, axes = plt.subplots(1, 3, figsize=(style.PAGE_W, 2.1))
fig.subplots_adjust(wspace=0.38)
for ax, (key, lab, logy) in zip(axes, panels):
    for fam in fams:
        for acc, sty in ((False, style.PLAIN), (True, style.SQUAREM)):
            if key != 'numiter' and not acc:
                continue
            med = []
            for ng in ngs:
                x = d[(d['family'] == fam) & (d['ngauss'] == ng)
                      & (d['accelerate'] == acc)][key]
                med.append(np.nanmedian(np.abs(x)))
            ax.plot(ngs, med, color=sty['color'], ls=sty['ls'],
                    marker=marks[fam], ms=3.5,
                    label=style.FAMILY_LABELS[fam] if acc else None)
    ax.set_xlabel('number of gaussians')
    ax.set_ylabel(lab)
    ax.set_xticks(ngs)
    if logy:
        ax.set_yscale('log')
axes[0].legend(loc='best', fontsize=6)
axes[2].text(0.97, 0.04, 'solid: EM\ndashed: SQUAREM', fontsize=6,
             transform=axes[2].transAxes, va='bottom', ha='right',
             color=style.REF)
fig.savefig(outdir + '/ngauss.pdf')
