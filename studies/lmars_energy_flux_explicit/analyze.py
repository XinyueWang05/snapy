#!/usr/bin/env python3
"""Tables for the explicit-path arms, the uniform-wind column and the box budgets.

Growth rate: least-squares slope of ln(rms w) vs t over the rows with
1e-4 <= rms w <= 1e-1 (>= 3 rows); otherwise the slope over t >= 0.5 d
(labelled 'run', a growth or decay rate). Writes tables.md and prints it.
"""
import csv, glob, json, math, os, sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(here, "runs")
DAY = 86400.0


def load(name):
    js = os.path.join(R, name + ".json")
    cs = os.path.join(R, name + ".csv")
    if not os.path.exists(cs):
        return None, None
    rows = list(csv.DictReader(open(cs)))
    rows = [{k: float(v) for k, v in r.items()} for r in rows]
    return (json.load(open(js)) if os.path.exists(js) else None), rows


def rate(rows):
    t = np.array([r["t"] for r in rows]); w = np.array([r["rms_w"] for r in rows])
    ok = np.isfinite(w) & (w > 0)
    band = ok & (w >= 1e-4) & (w <= 1e-1)
    if band.sum() >= 3:
        p = np.polyfit(t[band] / DAY, np.log(w[band]), 1)
        return p[0], "band", (t[band][0] / DAY, t[band][-1] / DAY)
    sel = ok & (t >= 0.5 * DAY)
    if sel.sum() >= 3:
        p = np.polyfit(t[sel] / DAY, np.log(w[sel]), 1)
        return p[0], "run", (t[sel][0] / DAY, t[sel][-1] / DAY)
    return float("nan"), "none", (None, None)


def at(rows, tday, key):
    best = min(rows, key=lambda r: abs(r["t"] - tday * DAY))
    if abs(best["t"] - tday * DAY) > 0.02 * DAY:
        return float("nan")
    return best[key]


def state(name, js, rows):
    if js:
        return js["status"], js["t_end_days"]
    st = os.path.join(R, name + ".state")
    return ("running" if os.path.exists(st) else "not started"), (rows[-1]["t"] / DAY if rows else 0.0)


def f(x, fmt="%.3g"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return fmt % x


out = []
out.append("## 1. Explicit path (implicit-scheme 0, lmars), nx2 = 64\n")
out.append("| arm | status | t_end (d) | dt median (s) | Mach | u dt/dx | cs dt/dz | rate (1/d) | fit | e-fold (dx/u) | lambda (dx) at rms w 1e-2 | lambda (dx) end | rms w end (m/s) |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
names = [l.split()[0] for l in open(os.path.join(here, "jobs.txt"))]
for n in [n for n in names if n.startswith("x_")]:
    js, rows = load(n)
    if rows is None:
        out.append(f"| {n} | not started |" + " |" * 11); continue
    st, te = state(n, js, rows)
    r, how, win = rate(rows)
    u = js["u0"] if js else float(n.split("_u")[1].split("_")[0])
    dx = js["dx"] if js else float(n.split("_dx")[1].split("_")[0]) * 1e3
    efold = (u / (r / DAY * dx)) if (u > 0 and r == r and r > 0) else float("nan")
    lm = js.get("lam_dx_mid") if js else None
    out.append("| %s | %s | %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        n, st, te, f(js and js["dt_median"], "%.4g"), f(js and js["mach"], "%.3f"), f(js and js["hcour"], "%.4f"),
        f(js and js["vcour"], "%.3f"), f(r, "%.2f"), how + ("" if win[0] is None else " %.2f-%.2f d" % win),
        f(efold, "%.2f"), f(lm[1], "%.1f") if lm else "n/a", f(js and js["lam_dx_end"], "%.1f"), f(rows[-1]["rms_w"], "%.2e")))

out.append("\n## 2. Uniform-wind column (nx2 = 4 identical periodic copies, no rotation or sponge)\n")
out.append("| arm | status | t_end (d) | dt median (s) | max|u-u0|/u0 @600 s | @1 d | @3 d | max over run | rms w @600 s | @1 d | @3 d (m/s) | rate (1/d) | fit |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for n in [n for n in names if n.startswith("col_")]:
    js, rows = load(n)
    if rows is None:
        out.append(f"| {n} | not started |" + " |" * 11); continue
    st, te = state(n, js, rows)
    r, how, win = rate(rows)
    mx = max(rr["max_du_rel"] for rr in rows[1:]) if len(rows) > 1 else float("nan")
    out.append("| %s | %s | %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        n, st, te, f(js and js["dt_median"], "%.4g"),
        f(at(rows, 600 / DAY, "max_du_rel"), "%.2e"), f(at(rows, 1, "max_du_rel"), "%.2e"), f(at(rows, 3, "max_du_rel"), "%.2e"), f(mx, "%.2e"),
        f(at(rows, 600 / DAY, "rms_w"), "%.2e"), f(at(rows, 1, "rms_w"), "%.2e"), f(at(rows, 3, "rms_w"), "%.2e"),
        f(r, "%.2f"), how + ("" if win[0] is None else " %.2f-%.2f d" % win)))
out.append("\nFor u0 = 0 the max|u-u0|/u0 columns hold max|u| in m/s.\n")

out.append("\n## 3. Periodic box budgets, u = 400 m/s, dx = 100 km (relative change from t = 0 to t_end; max |change| over the run in brackets)\n")
out.append("| arm | status | t_end (d) | redos | rho | m2 | m3 / (S_rho cs) | E (no PE) | E + rho g z | rate (1/d) | fit | e-fold (dx/u) | lambda (dx) at rms w 1e-2 | rms w end | rms(u-u0) end |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for n in ["box_dx100_u400_s9", "box_dx100_u400_s1", "x_dx100_u400_s0", "x_dx100_u0_s0_nonoise"]:
    js, rows = load(n)
    if rows is None:
        out.append(f"| {n} | not started |" + " |" * 7); continue
    st, te = state(n, js, rows)
    r0 = rows[0]
    def rel(k):
        d = [(rr[k] - r0[k]) / abs(r0[k]) for rr in rows]
        return "%.2e [%.2e]" % (d[-1], max(abs(x) for x in d))
    cs = js["cs"] if js else 742.0
    m3 = [rr["S_m3"] / (r0["S_rho"] * cs) for rr in rows]
    m2 = rel("S_m2") if r0["S_m2"] != 0 else "%.2e (abs, kg m/s)" % rows[-1]["S_m2"]
    r, how, win = rate(rows)
    u, dx = (js["u0"], js["dx"]) if js else (0.0, 1e5)
    efold = (u / (r / DAY * dx)) if (u > 0 and r == r and r > 0) else float("nan")
    lm = js.get("lam_dx_mid") if js else None
    out.append("| %s | %s | %.3f | %s | %s | %s | %.2e [%.2e] | %s | %s | %s | %s | %s | %s | %.2e | %.2e |" % (
        n, st, te, js["redos"] if js else "n/a", rel("S_rho"), m2, m3[-1], max(abs(x) for x in m3), rel("S_E"), rel("S_Etot"),
        f(r, "%.2f"), how + ("" if win[0] is None else " %.2f-%.2f d" % win), f(efold, "%.2f"),
        f(lm[1], "%.1f") if lm else "n/a", rows[-1]["rms_w"], rows[-1]["rms_du"]))
txt = "\n".join(out) + "\n"
open(os.path.join(here, "tables.md"), "w").write(txt)
print(txt)
