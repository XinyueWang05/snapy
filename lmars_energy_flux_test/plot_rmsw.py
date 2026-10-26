"""Plot rms w(t) of every arm: one panel per dx, lmars solid, hllc dashed,
colour by u; optional extra run dirs (e.g. the break test) drawn in black.

usage: python plot_rmsw.py <runs dir> <out.png> [<extra run dir> <label>] ...
"""
import glob, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

runs, png = sys.argv[1], sys.argv[2]
extra = list(zip(sys.argv[3::2], sys.argv[4::2]))
col = {0.0: "C0", 50.0: "C1", 250.0: "C2", 400.0: "C3"}
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, dx in zip(axs, (5, 50, 400)):
    for d in sorted(glob.glob(os.path.join(runs, "*_dx%d" % dx))):
        s = np.genfromtxt(os.path.join(d, "series.csv"), delimiter=",", names=True)
        tag = os.path.basename(d)
        solver, u = tag.split("_")[0], float(tag.split("_u")[1].split("_")[0])
        ax.semilogy(s["t"] / 86400, np.maximum(s["rms_w"], 1e-16), color=col[u],
                    ls="-" if solver == "lmars" else "--", label="%s u=%g" % (solver, u))
    for d, lab in extra:
        if d.endswith("_dx%d" % dx) or ("dx%d" % dx) in os.path.basename(d.rstrip("/")):
            s = np.genfromtxt(os.path.join(d, "series.csv"), delimiter=",", names=True)
            ax.semilogy(s["t"] / 86400, s["rms_w"], color="k", lw=1, label=lab)
    ax.set_title("dx = %d km (64 columns)" % dx)
    ax.set_xlabel("t [day]")
    ax.grid(alpha=0.3)
axs[0].set_ylabel("rms w [m/s]")
axs[-1].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(png, dpi=150)
print(png)
