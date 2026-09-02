"""Build Kilosort/probeinterface probe files for the two acute Cambridge NeuroTech
ASSY-107 128-channel probes, transcribed from the CN channel-map printouts
(IMG_1041 = H5-D, IMG_1042 = H15; 'Last updated 03 Mar 2026').

Channel numbers = 0-indexed Intan/Open Ephys software channels (per the printout).
Geometry: each shank/face uses the CN H5 two-column staggered layout (x cols 22.5 um
apart, 25 um within-column pitch, 12.5 um stagger -> 788 um span, verified against
probeinterface ASSY-77-H5). Double-sided front face offset +45 um in x; H15 shank B
offset +250 um. Those offsets are drawing conventions; the CHANNEL MAP is exact.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from probeinterface import Probe, write_probeinterface

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- transcribed channel maps (top->bottom of each table column) ----
H5D = dict(
    back_left =[15,13,11,9,6,8,3,1,64,2,4,7,5,10,12,14,17,40,36,32,33,37,41,45,50,54,55,62,61,56,53,49],
    back_right=[19,21,23,25,27,29,31,80,30,28,26,24,22,20,18,46,44,42,38,34,111,35,39,43,48,52,57,60,127,59,58,51],
    front_left=[112,114,116,118,120,126,124,122,63,123,125,121,119,117,115,113,96,89,85,81,82,86,90,94,77,73,65,69,68,70,74,78],
    front_right=[98,100,102,104,106,108,110,47,109,107,105,103,101,99,97,95,93,91,87,83,16,84,88,92,79,75,71,67,0,66,72,76],
)
H15 = dict(
    A_left =[95,93,91,89,87,85,83,81,16,82,84,86,88,90,92,94,79,9,8,1,2,7,10,14,19,23,27,31,30,26,22,18],
    A_right=[77,75,73,71,65,67,69,0,68,66,70,72,74,76,78,15,13,11,6,3,64,4,5,12,17,21,25,29,80,28,24,20],
    B_left =[46,44,42,40,38,36,34,32,111,33,35,37,39,41,43,45,48,118,126,122,123,121,117,113,98,102,106,110,109,105,101,97],
    B_right=[50,52,54,57,55,60,62,127,61,59,56,58,53,51,49,112,114,116,120,124,63,125,119,115,96,100,104,108,47,107,103,99],
)

YPITCH = 25.0        # within-column vertical pitch (um)
XCOL   = 22.5        # column separation (um)
STAG   = 12.5        # left/right stagger (um)

def col_positions(x0, n=32):
    """left column x=x0 (y = (31-i)*25), right column x=x0+22.5 (y +12.5)."""
    i = np.arange(n)
    yL = (n-1-i)*YPITCH
    yR = (n-1-i)*YPITCH + STAG
    return (np.full(n, x0), yL), (np.full(n, x0+XCOL), yR)

def build(groups, shank_of):
    xs, ys, chan, shanks = [], [], [], []
    for (x0, sh, Lkey, Rkey), g in groups:
        (xL,yL),(xR,yR) = col_positions(x0)
        xs += list(xL)+list(xR); ys += list(yL)+list(yR)
        chan += list(g[Lkey])+list(g[Rkey]); shanks += [sh]*64
    return np.array(xs,float), np.array(ys,float), np.array(chan,int), np.array(shanks,int)

# H5-D: single shank, back face (x 0/22.5) + front face (x 45/67.5)
x,y,chan,sh = build([((0.0,0,'back_left','back_right'), H5D),
                     ((45.0,0,'front_left','front_right'), H5D)], None)
def make_probe(x,y,chan,sh,name):
    p = Probe(ndim=2, si_units='um')
    p.set_contacts(positions=np.c_[x,y], shapes='rect',
                   shape_params={'width':11,'height':15})
    p.set_device_channel_indices(chan)     # contact i is recorded by software channel chan[i]
    p.annotate(name=name)
    if sh is not None: p.set_shank_ids(sh.astype(str))
    return p

p_h5d = make_probe(x,y,chan,sh,'ASSY-107_H5-D')
write_probeinterface(os.path.join(OUT,'probe_H5D.json'), p_h5d)

# H15: two shanks single-sided (A x 0/22.5 shank0, B x 250/272.5 shank1)
x2,y2,chan2,sh2 = build([((0.0,0,'A_left','A_right'), H15),
                         ((250.0,1,'B_left','B_right'), H15)], None)
p_h15 = make_probe(x2,y2,chan2,sh2,'ASSY-107_H15')
write_probeinterface(os.path.join(OUT,'probe_H15.json'), p_h15)

print('wrote probe_H5D.json (128 ch, 1 shank, double-sided)')
print('wrote probe_H15.json (128 ch, 2 shanks)')

# ---- verification figure: reproduce each printout's table layout ----
def table_fig(groups, title, fname):
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2*len(groups), 12))
    if len(groups)==1: axes=[axes]
    for ax,(label,Lkey,Rkey,g) in zip(axes, groups):
        L,R = g[Lkey], g[Rkey]
        ax.set_title(label, fontsize=13, fontweight='bold')
        for row in range(33):
            # left col occupies rows 1..32, right col rows 0..31 (stagger, as printed)
            if 1<=row<=32: ax.text(0.35, 32-row, str(L[row-1]), ha='center', va='center', fontsize=10)
            if 0<=row<=31: ax.text(0.65, 32-row-0.5, str(R[row]), ha='center', va='center', fontsize=10)
        ax.set_xlim(0,1); ax.set_ylim(-1,34); ax.axis('off')
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,fname), dpi=130, bbox_inches='tight'); plt.close(fig)

table_fig([('Shank A (back)','back_left','back_right',H5D),
           ('Shank A (front)','front_left','front_right',H5D)],
          'ASSY-107 H5-D  — my transcription (compare to printout)', 'verify_H5D.png')
table_fig([('Shank A','A_left','A_right',H15),
           ('Shank B','B_left','B_right',H15)],
          'ASSY-107 H15  — my transcription (compare to printout)', 'verify_H15.png')
print('wrote verify_H5D.png, verify_H15.png')
