"""
Shared figure style.  Colours are slots 1 and 2 of a categorical palette
validated for colour-vision deficiency (blue/orange); every series also has
its own line style, so no figure depends on colour alone.
"""
import matplotlib as mpl

COL_W = 3.4    # ApJ single column, inches
PAGE_W = 7.1   # ApJ full width

PLAIN = dict(color='#2a78d6', ls='-', label='EM')
SQUAREM = dict(color='#eb6834', ls='--', label='SQUAREM')
REF = '#52514e'          # reference lines and annotations
FAMILY_LABELS = {
    'superbit': 'SuperBIT PSFEx',
    'hst': 'HST/ACS (COSMOS)',
    'roman': 'Roman H158',
}
FAMILIES = list(FAMILY_LABELS)
NGAUSS_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']  # 3, 5, 7, 9

TOL_SUPERBIT = 1.0e-6
TOL_DEFAULT = 1.0e-5


def setup():
    mpl.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'lines.linewidth': 1.4,
        'legend.frameon': False,
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
    })
