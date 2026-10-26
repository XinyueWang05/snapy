"""Growth rate of rms w vs |u|/dx and vs Mach number across all arms.

Reads the per-arm CSVs written by analyze.py (one per run set, e.g. the
original T0 = 100 K matrix, the hot-Jupiter matrix and the u* pbar break test)
and draws two panels. lmars and hllc have distinct colours, the setup
(T0 = 100 K / 1500 K) distinct marker shapes, the break test is a cross.
Arms with no exponential growth ('none' in analyze.py) are drawn as open
markers on a floor line, u = 0 arms at the left edge of the |u|/dx panel.

usage: python plot_growth.py <out.png> <arms.csv> [<arms.csv> ...] [--break <break.csv>]...
"""
import csv, sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLOOR = 3.0e-6  # [1/s] y position of non-growing arms
X0 = 3.0e-5  # [1/s] x position of u = 0 arms in the |u|/dx panel

args = sys.argv[1:]
out = args.pop(0)
brk = []
while "--break" in args:
    i = args.index("--break")
    brk.append(args[i + 1])
    del args[i:i + 2]


def read(paths):
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    return rows


def g(r):
    return float(r["growth_per_s"]) if r["growth_per_s"] not in ("", "None") else None


color = {"lmars": "tab:red", "hllc": "tab:blue"}
shape = {100.0: "o", 1500.0: "^"}
fig, axs = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
seen = set()
for rows, is_break in ((read(args), False), (read(brk), True)):
    for r in rows:
        k, T0, s = g(r), float(r["T0"]), r["solver"]
        y = k if k else FLOOR
        xs = (float(r["u_over_dx"]) or X0, float(r["mach"]))
        if is_break:
            lab = "lmars u* pbar break test, T0 = %g K" % T0
            kw = dict(marker="x", color="k", s=70, zorder=5)
        else:
            lab = "%s, T0 = %g K%s" % (s, T0, "" if k else " (no growth)")
            m = shape.get(T0, "s")
            kw = dict(marker=m, s=40 if s == "lmars" else 90, zorder=4 if s == "lmars" else 3,
                      facecolors=color[s] if k else "none", edgecolors=color[s])
        for ax, x in zip(axs, xs):
            ax.scatter([x], [y], label=lab if ax is axs[1] and lab not in seen else None, **kw)
        seen.add(lab)
for ax in axs:
    ax.axhline(FLOOR, color="0.6", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
axs[0].set_xscale("log")
axs[0].axvline(X0 * 1.6, color="0.6", lw=0.8, ls=":")
axs[0].text(X0, FLOOR * 1.5, "u=0", ha="center", fontsize=8, color="0.4")
axs[0].set_xlabel("|u| / dx  [1/s]")
axs[1].set_xlabel("Mach number |u| / c_s")
axs[0].set_ylabel("exponential growth rate of rms w  [1/s]")
axs[1].text(0.98, FLOOR * 1.15, "floor: no exponential growth", ha="right", va="bottom",
            fontsize=8, color="0.4", transform=axs[1].get_yaxis_transform())
fig.legend(*axs[1].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=3,
           frameon=False)
fig.suptitle("snapy a09cc256, 2-D dry H2-He, weno5/rk3/implicit-scheme 9, 3 model days: "
             "rms w growth rate, all arms", fontsize=10)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(out, dpi=150)
print("wrote", out)
