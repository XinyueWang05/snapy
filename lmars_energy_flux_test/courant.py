"""Growth/decay rate of rms w per arm against u, dx, u*dt/dx and cs*dt/dz.

For every <runs>/<set>/<arm>/ (series.csv + summary.json):
  dt        : dt_max from summary.json, the step of the unperturbed flow
              (the per-hour clamp to the diagnostic time and the saturated
              phase only shorten it); dt_mean = 3 d / cycles is also listed
  Courant   : u*dt/dx (horizontal), cs*dt/dz (vertical acoustic),
              cs = sqrt(1.4 Rd T0), dz = ztop / nx1, Mach = u / cs
  rate      : linear fit of ln(rms w) in time, in 1/day.
              growing arm (rms w rises >= 10x from its minimum after 6 h):
                window = [argmin after 6 h, first sample with
                rms w >= 1e-2 * max rms w]  (exponential phase, ending two
                decades below saturation)
                (needs >= 6 samples and R^2 >= 0.9)
              otherwise: window = [1 d, end] (runs of >= 2 d) or
                [0.25 d, end] (1-day diagnostics); 'decays' if rate < 0 and
                R^2 >= 0.5, else 'no trend'
  lambda/dx : dominant x2 wavelength of w at 1 d and at the end (analyze.py)

usage: python courant.py <runs root> <out.md> <out.csv> <set> [<set> ...]
"""
import csv, json, os, sys, glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DAY = 86400.0
GAMMA = 1.4
root, out_md, out_csv, sets = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]


def wavelength(d, nx2, tag):
    z = np.load(os.path.join(d, "snaps.npz"))
    tags = [k[:-2] for k in z.files if k.endswith("_t")]
    if tag == "end":
        tag = "stop" if "stop" in tags else max(tags, key=lambda k: float(z[k + "_t"]))
    if tag not in tags:
        return np.nan
    w = z[tag + "_w"]
    lev = int(np.argmax(np.sqrt(((w - w.mean(0)) ** 2).mean(0))))
    a = np.abs(np.fft.rfft(w[:, lev] - w[:, lev].mean()))
    return nx2 / (1 + int(np.argmax(a[1:])))


def fit(t, r):
    x, y = t / DAY, np.log(r)
    k, b = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (k * x + b)) ** 2) / np.sum((y - y.mean()) ** 2)
    return k, r2


rows = []
for st in sets:
    for d in sorted(glob.glob(os.path.join(root, st, "*"))):
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        sm = json.load(open(os.path.join(d, "summary.json")))
        s = np.atleast_1d(np.genfromtxt(os.path.join(d, "series.csv"), delimiter=",", names=True))
        t, r = s["t"], s["rms_w"]
        z0 = np.load(os.path.join(d, "snaps.npz"))
        T0 = sm.get("T0") or float(round(z0["d0_T"].mean()))
        ztop = sm.get("ztop") or (450e3 if T0 == 100 else 2570e3)
        cs = np.sqrt(GAMMA * sm["Rd"] * T0)
        dz = ztop / sm["nx1"]
        u, dx, dt = abs(sm["u"]), sm["dx"], sm["dt_max"]
        ok = (t >= 6 * 3600) & (r > 0)
        tt, rr = t[ok], r[ok]
        i0 = int(np.argmin(rr))
        grows = rr[i0:].max() / rr[i0] >= 10
        if grows:
            i1 = i0 + int(np.argmax(rr[i0:] >= 1e-2 * rr.max()))
            if i1 - i0 + 1 < 6:
                grows = False
            else:
                k, r2 = fit(tt[i0:i1 + 1], rr[i0:i1 + 1])
                grows = r2 >= 0.9
        if grows:
            win, cls = (tt[i0] / DAY, tt[i1] / DAY), "grows"
        else:
            t1 = DAY if t[-1] >= 2 * DAY else 0.25 * DAY
            m = (t >= t1) & (r > 0)
            k, r2 = fit(t[m], r[m])
            win = (t[m][0] / DAY, t[m][-1] / DAY)
            cls = "decays" if (k < 0 and r2 >= 0.5) else "no trend"
        rows.append({"set": st, "arm": os.path.basename(d), "solver": sm["solver"],
                     "scheme": sm["scheme"], "T0": T0, "u": u, "dx_km": dx / 1e3,
                     "dz_km": dz / 1e3, "cs": cs, "mach": u / cs, "dt_s": dt,
                     "dt_mean_s": sm["t_end"] / sm["cycles"], "u_dt_dx": u * dt / dx,
                     "cs_dt_dz": cs * dt / dz, "class": cls, "rate_per_day": k,
                     "fit_from_d": win[0], "fit_to_d": win[1], "r2": r2,
                     "lambda_dx_1d": wavelength(d, sm["nx2"], "d1"),
                     "lambda_dx_end": wavelength(d, sm["nx2"], "end"),
                     "max_w": float(np.nanmax(s["max_w"])), "rms_w_end": float(r[-1]),
                     "t_end_d": sm["t_end"] / DAY, "status": sm["status"]})

order = {"lmars": 0, "hllc": 1}
rows.sort(key=lambda q: (q["set"], q["T0"], -q["dx_km"], q["u"], order[q["solver"]]))
hdr = ("| set | solver | u | dx km | Mach | dt s | u dt/dx | cs dt/dz | class | rate 1/day | "
       "fit window d | R^2 | lambda/dx 1 d | lambda/dx end | max abs w | status |")
L = [hdr, "|" + "---|" * 16]
for q in rows:
    L.append("| %s | %s | %g | %g | %.3f | %.3g | %.4f | %.3f | %s | %+.3g | %.2f-%.2f | %.3f | %.1f | %.1f | %.2e | %s (%.2f d) |" % (
        q["set"], q["solver"], q["u"], q["dx_km"], q["mach"], q["dt_s"], q["u_dt_dx"],
        q["cs_dt_dz"], q["class"], q["rate_per_day"], q["fit_from_d"], q["fit_to_d"], q["r2"],
        q["lambda_dx_1d"], q["lambda_dx_end"], q["max_w"], q["status"], q["t_end_d"]))
txt = "\n".join(L)
print(txt)
open(out_md, "w").write(txt + "\n")
with open(out_csv, "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
