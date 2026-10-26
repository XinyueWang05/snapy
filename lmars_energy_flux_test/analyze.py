"""Summarize LMARS energy-flux arms.

For every <runs>/<tag>/ with series.csv: rms w at t = 0, 1, 2, 3 d, max |w|
over the run, the exponential growth rate of rms w and the dominant
horizontal wavelength of w in units of dx.

Growth rate: log(rms w) is fitted linearly on the growth phase, defined as
the span from the minimum of rms w after the first 6 h to the maximum of rms w
that follows it. A rate is reported only if that span covers >= 6 samples, rms
w grows by >= 10x across it and the fit has R^2 >= 0.9; otherwise 'none'.

Wavelength: from the last w snapshot (t = 3 d, or the stop time), at the x1
level with the largest rms of w along x2, the FFT of w - mean along x2; the
dominant wavenumber m >= 1 gives lambda/dx = nx2/m. lambda_dx_d1 is the same
from the t = 1 d snapshot (inside the exponential phase of the growing arms).

Mach = |u|/c_s with c_s = sqrt(gamma Rd T0), gamma = 1 + 1/cv_R = 1.4 for this
deck (Rd and T0 from summary.json; for runs that predate the T0 field, T0 is the
mean of the t = 0 T snapshot, rounded).

usage: python analyze.py <runs dir> [<out.md>] [<out.csv>]
"""
import csv, glob, json, os, sys

import numpy as np

DAY = 86400.0
GAMMA = 1.4  # cv_R = 2.5
RD = 3936.0434480313606  # from the deck EOS (run_arm.py)
runs = sys.argv[1]
out_md = sys.argv[2] if len(sys.argv) > 2 else None
out_csv = sys.argv[3] if len(sys.argv) > 3 else None


def load(d):
    a = np.genfromtxt(os.path.join(d, "series.csv"), delimiter=",", names=True)
    return np.atleast_1d(a)


def at(s, tday):
    i = np.where(np.abs(s["t"] - tday * DAY) < 1.0)[0]
    return s["rms_w"][i[0]] if len(i) else np.nan


def growth(s):
    t, r = s["t"], s["rms_w"]
    ok = (t >= 6 * 3600) & (r > 0) & np.isfinite(r)
    if ok.sum() < 6:
        return None, None
    t, r = t[ok], r[ok]
    i0 = int(np.argmin(r))
    i1 = i0 + int(np.argmax(r[i0:]))
    if i1 - i0 + 1 < 6 or r[i1] / r[i0] < 10:
        return None, (t[i0], t[i1], r[i1] / r[i0])
    x, y = t[i0:i1 + 1], np.log(r[i0:i1 + 1])
    k, b = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (k * x + b)) ** 2) / np.sum((y - y.mean()) ** 2)
    if r2 < 0.9:
        return None, (t[i0], t[i1], r[i1] / r[i0], r2)
    return k, (t[i0], t[i1], r[i1] / r[i0], r2)


def wavelength(d, nx2, tag=None):
    z = np.load(os.path.join(d, "snaps.npz"))
    tags = [k[:-2] for k in z.files if k.endswith("_t")]
    if tag is not None and tag not in tags:
        return np.nan, -1, np.nan, np.nan
    tag = tag or ("stop" if "stop" in tags else max(tags, key=lambda k: float(z[k + "_t"])))
    w = z[tag + "_w"]  # (nx2, nx1)
    lev = int(np.argmax(np.sqrt(((w - w.mean(0)) ** 2).mean(0))))
    a = np.abs(np.fft.rfft(w[:, lev] - w[:, lev].mean()))
    m = 1 + int(np.argmax(a[1:]))
    return nx2 / m, lev, float(z[tag + "_t"]), float(z["x1"][lev])


rows = []
for d in sorted(glob.glob(os.path.join(runs, "*"))):
    if not os.path.exists(os.path.join(d, "series.csv")):
        continue
    s = load(d)
    sm = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(
        os.path.join(d, "summary.json")) else {}
    tag = os.path.basename(d)
    solver, u, dx = tag.split("_")[0], float(tag.split("_u")[1].split("_")[0]), \
        float(tag.split("_dx")[1].split("_")[0])
    k, info = growth(s)
    nx2 = sm.get("nx2", 64)
    lam, lev, tl, zl = wavelength(d, nx2) if os.path.exists(os.path.join(d, "snaps.npz")) \
        else (np.nan, -1, np.nan, np.nan)
    T0 = sm.get("T0")
    if T0 is None:  # older summaries: isothermal T0 from the t = 0 snapshot
        T0 = float(round(np.load(os.path.join(d, "snaps.npz"))["d0_T"].mean()))
    cs = np.sqrt(GAMMA * sm.get("Rd", RD) * T0)
    lam1 = wavelength(d, nx2, "d1")[0] if os.path.exists(os.path.join(d, "snaps.npz")) \
        else np.nan
    rows.append({"arm": tag, "solver": solver, "u": u, "dx_km": dx, "T0": T0,
                 "mach": abs(u) / cs, "u_over_dx": abs(u) / (dx * 1e3),
                 "rms_w_0d": at(s, 0), "rms_w_1d": at(s, 1), "rms_w_2d": at(s, 2),
                 "rms_w_3d": at(s, 3), "max_w": float(np.nanmax(s["max_w"])),
                 "growth_per_s": k, "efold_h": (1 / k / 3600 if k else None),
                 "growth_window_d": (None if info is None else
                                     "%.2f-%.2f" % (info[0] / DAY, info[1] / DAY)),
                 "growth_factor": None if info is None else info[2],
                 "lambda_dx": lam, "lambda_dx_d1": lam1, "level_km": zl / 1e3 if zl == zl else None,
                 "t_last_d": float(s["t"][-1] / DAY), "status": sm.get("status", "running"),
                 "wall_s_per_model_h": sm.get("wall_s_per_model_hour",
                                              s["wall"][-1] / max(s["t"][-1] / 3600, 1e-30))})

order = {"lmars": 0, "hllc": 1}
rows.sort(key=lambda r: (r["T0"], order.get(r["solver"], 2), -r["dx_km"], r["u"]))


def f(x, fmt="%.2e"):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt % x


hdr = ("| solver | T0 [K] | u [m/s] | dx [km] | Mach | abs(u)/dx [1/s] | rms w 0 d | 1 d | 2 d | 3 d | max abs w | "
       "growth [1/s] (e-fold) | lambda/dx (end) | lambda/dx (1 d) | status (t_end d) |")
lines = [hdr, "|" + "---|" * 15]
for r in rows:
    g = "none" if r["growth_per_s"] is None else "%.2e (%.1f h)" % (r["growth_per_s"], r["efold_h"])
    lines.append("| %s | %g | %g | %g | %.3f | %.1e | %s | %s | %s | %s | %s | %s | %s | %s | %s (%.2f) |" % (
        r["solver"], r["T0"], r["u"], r["dx_km"], r["mach"], r["u_over_dx"], f(r["rms_w_0d"]), f(r["rms_w_1d"]), f(r["rms_w_2d"]),
        f(r["rms_w_3d"]), f(r["max_w"]), g, f(r["lambda_dx"], "%.1f"), f(r["lambda_dx_d1"], "%.1f"), r["status"], r["t_last_d"]))
txt = "\n".join(lines)
print(txt)
if out_md:
    open(out_md, "w").write(txt + "\n")
if out_csv:
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
