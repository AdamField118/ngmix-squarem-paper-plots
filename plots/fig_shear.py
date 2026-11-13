"""
Shear calibration for each arm relative to the fully converged em5 fit:
paired differences in m, c, PSF leakage alpha and the response R11, for every
PSF family.  Reads the summary.json files written by
squarem_shear.collect (https://github.com/AdamField118/ngmix-squarem-shear).

    python plots/fig_shear.py figures/ data/shear/*/summary.json
"""
import sys
import json
import os
import matplotlib.pyplot as plt
import style

style.setup()
outdir = sys.argv[1]
files = sys.argv[2:]
ARMS = [('em5_default', 'EM, ngmix default', style.PLAIN, 'o'),
        ('em5_superbit', 'EM, SuperBIT', style.PLAIN, 's'),
        ('em5_squarem', 'SQUAREM, SuperBIT tol', style.SQUAREM, 'D')]
QUANT = [('dm', r'$\Delta m$'), ('dc1', r'$\Delta c_1$'),
         ('dalpha', r'$\Delta \alpha$'), ('dR11', r'$\Delta R_{11}$')]

sums = []
for f in files:
    with open(f) as fh:
        sums.append((os.path.basename(os.path.dirname(f)), json.load(fh)))
if not sums:
    # keep the paper compilable before the shear runs exist
    fig, ax = plt.subplots(figsize=(style.PAGE_W, 2.2))
    ax.text(0.5, 0.5, 'shear results not yet generated', ha='center',
            va='center', color=style.REF)
    ax.set_axis_off()
    fig.savefig(outdir + '/shear_bias.pdf')
    sys.exit(0)
order = {f: i for i, f in enumerate(style.FAMILIES)}
sums.sort(key=lambda x: order.get(x[0], 99))

fig, axes = plt.subplots(1, len(QUANT), figsize=(style.PAGE_W, 2.2),
                         sharey=True)
for ax, (key, lab) in zip(axes, QUANT):
    for j, (name, s) in enumerate(sums):
        for k, (arm, alab, sty, mk) in enumerate(ARMS):
            v = s['arms'].get(arm)
            if v is None:
                continue
            y = j + (k - 1) * 0.22
            ax.errorbar(v[key], y, xerr=v[key + '_err'], marker=mk, ms=3.5,
                        color=sty['color'], ls='none', elinewidth=0.9,
                        label=alab if j == 0 else None)
    ax.axvline(0, color=style.REF, ls=':', lw=0.8)
    ax.set_xlabel(lab + '  (arm $-$ converged)')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
axes[0].set_yticks(range(len(sums)))
axes[0].set_yticklabels([style.FAMILY_LABELS.get(n, n) for n, _ in sums])
axes[0].invert_yaxis()
axes[-1].legend(loc='lower right', fontsize=6)
if min(s['ngal'] for _, s in sums) < 10000:
    # make an under-powered draft impossible to mistake for the result
    fig.text(0.5, 0.5, 'PRELIMINARY: %d galaxies' % min(
        s['ngal'] for _, s in sums), ha='center', va='center', fontsize=20,
        color='#e34948', alpha=0.35, rotation=12)
fig.savefig(outdir + '/shear_bias.pdf')
