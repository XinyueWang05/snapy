"""Growth/decay rate of rms w (1/day) vs u*dt/dx and vs cs*dt/dz, all arms.

Reads the courant.py CSV. Filled markers: exponential growth; open markers:
no growth (rate fitted over 1-3 d, decay or no trend). Colour = solver,
shape = deck (T0 100 K / 1500 K); break tests and scheme-0 diagnostics are
left out (same dt as their parent arm).

usage: python plot_courant.py <courant.csv> <out.png>
"""
import csv, sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["set"] in ("matrix", "hj")]
color = {"lmars": "tab:red", "hllc": "tab:blue"}
shape = {100.0: "o", 1500.0: "^"}
fig, axs = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
seen = set()
for r in rows:
    k, s, T0 = float(r["rate_per_day"]), r["solver"], float(r["T0"])
    g = r["class"] == "grows"
    lab = "%s, T0 = %g K%s" % (s, T0, "" if g else " (no growth)")
    kw = dict(marker=shape[T0], s=40 if s == "lmars" else 90, zorder=4 if s == "lmars" else 3,
              facecolors=color[s] if g else "none", edgecolors=color[s])
    for ax, x in zip(axs, (float(r["u_dt_dx"]), float(r["cs_dt_dz"]))):
        ax.scatter([x], [k], label=lab if ax is axs[1] and lab not in seen else None, **kw)
    seen.add(lab)
for ax in axs:
    ax.axhline(0, color="0.5", lw=0.8)
    ax.grid(alpha=0.3, which="both")
axs[1].set_xscale("log")
axs[0].set_xlabel("horizontal Courant number  |u| dt / dx")
axs[1].set_xlabel("vertical acoustic Courant number  c_s dt / dz")
axs[0].set_ylabel("rms w growth rate  [1/day]  (negative: decay)")
fig.legend(*axs[1].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=4,
           frameon=False)
fig.suptitle("snapy a09cc256, 2-D dry H2-He, weno5/rk3/implicit-scheme 9, cfl 0.3, 3 model days",
             fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(sys.argv[2], dpi=150)
print("wrote", sys.argv[2])
