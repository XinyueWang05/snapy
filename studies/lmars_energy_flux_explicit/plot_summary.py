#!/usr/bin/env python3
"""Summary figure: rms w vs time (box arms, explicit arms) and max|u-u0|/u0 (column)."""
import csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(here, "runs")


def rows(n):
    r = list(csv.DictReader(open(os.path.join(R, n + ".csv"))))
    return [float(x["t"]) / 86400 for x in r], {k: [float(x[k]) for x in r] for k in r[0]}


fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for n in sorted(glob.glob(os.path.join(R, "x_dx*_s0.csv"))):
    n = os.path.basename(n)[:-4]
    t, d = rows(n)
    ax[0].semilogy(t[1:], d["rms_w"][1:], color="0.7", lw=0.6)
for n, c in (("box_dx100_u400_s9", "C0"), ("box_dx100_u400_s1", "C3"), ("x_dx100_u400_s0", "k")):
    t, d = rows(n)
    ax[0].semilogy(t[1:], d["rms_w"][1:], color=c, lw=1.5, label=n)
ax[0].set_xlabel("t (day)"); ax[0].set_ylabel("rms w (m/s)")
ax[0].set_title("2-D box, u = 400 m/s, dx = 100 km (grey: all 16 explicit arms)")
ax[0].legend(fontsize=8)
for s, c in (("0", "k"), ("1", "C3"), ("9", "C0")):
    for u, ls in (("100", ":"), ("400", "-")):
        t, d = rows(f"col_u{u}_s{s}")
        ax[1].semilogy(t[1:], [max(v, 1e-17) for v in d["max_du_rel"][1:]], color=c, ls=ls, label=f"scheme {s}, u {u}")
ax[1].set_xlabel("t (day)"); ax[1].set_ylabel("max|u-u0|/u0")
ax[1].set_title("uniform-wind column")
ax[1].legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(here, "summary.png"), dpi=150)
