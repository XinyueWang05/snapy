#!/usr/bin/env python3
"""Before/after tables and summary figure for the scheme-1 donor-momentum fix.

usage: analyze.py <runs dir> <out.md> <out.png>
Rate: slope of ln(rms w) vs t (1/day) over rows with 1e-4 <= rms w <= 1e-1
(>= 3 rows, 'band', same definition as the explicit-path study), else over
t >= 0.5 d ('run'). Drift: max over the run of |S(t) - S(0)| / |S(0)|.
"""
import csv, hashlib, json, math, os, sys

import numpy as np

DAY = 86400.0
R, OUT_MD, OUT_PNG = sys.argv[1:4]


def load(tree, name):
    p = os.path.join(R, tree, name)
    if not os.path.exists(p + ".json"):
        return None
    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(p + ".csv"))]
    js = json.load(open(p + ".json"))
    js["rows"] = rows
    js["csv_sha"] = hashlib.sha256(open(p + ".csv", "rb").read()).hexdigest()
    return js


def rate(rows):
    t = np.array([r["t"] for r in rows]) / DAY
    w = np.array([r["rms_w"] for r in rows])
    ok = np.isfinite(w) & (w > 0)
    band = ok & (w >= 1e-4) & (w <= 1e-1)
    if band.sum() >= 3:
        return np.polyfit(t[band], np.log(w[band]), 1)[0], "band %.2f-%.2f d" % (t[band][0], t[band][-1])
    sel = ok & (t >= 0.5)
    if sel.sum() < 3:
        return float("nan"), "rms w = 0"
    return np.polyfit(t[sel], np.log(w[sel]), 1)[0], "run 0.50-%.2f d" % t[sel][-1]


def rate_window(rows, a=0.5, b=3.0):
    t = np.array([r["t"] for r in rows]) / DAY
    w = np.array([r["rms_w"] for r in rows])
    sel = np.isfinite(w) & (w > 0) & (t >= a) & (t <= b)
    return np.polyfit(t[sel], np.log(w[sel]), 1)[0] if sel.sum() >= 3 else float("nan")


def drift(rows, key):
    s = np.array([r[key] for r in rows])
    if s[0] == 0:
        return float(np.abs(s).max()), "abs"
    return float(np.abs((s - s[0]) / s[0]).max()), "rel"


names = sorted({f[:-5] for tree in ("base", "fix") if os.path.isdir(os.path.join(R, tree))
                for f in os.listdir(os.path.join(R, tree)) if f.endswith(".json")},
               key=lambda n: (("u0" in n), int(n.split("_dx")[1].split("_")[0]), n))
md = ["| arm | tree | status | steps | redos | rate (1/d) | fit | rate 0.5-3 d (1/d) | rms w end (m/s) | "
      "rms(u-u0) end | drift rho | drift m2 | m3 | drift E (no PE) | drift E+rho g z |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
res = {}
for n in names:
    for tree in ("base", "fix"):
        d = load(tree, n)
        if d is None:
            continue
        res[(n, tree)] = d
        r, fit = rate(d["rows"])
        dr = {k: drift(d["rows"], k) for k in ("S_rho", "S_m2", "S_m3", "S_E", "S_Etot")}
        d["rate"], d["fit"], d["drift"] = r, fit, dr
        d["rate_w"] = rate_window(d["rows"])
        last = d["rows"][-1]
        # zero initial sum (m2 at u0 = 0, m3): absolute drift / (S_rho c_s), c_s = 742.34 m/s
        norm = d["rows"][0]["S_rho"] * d["cs"]
        f = lambda x: "%.2e" % x[0] if x[1] == "rel" else ("%.2e (/S_rho c_s)" % (x[0] / norm) if x[0] else "0")
        md.append("| %s | %s | %s | %d | %d | %+.2f | %s | %+.2f | %.3g | %.3g | %s | %s | %s | %s | %s |" % (
            n, tree, d["status"], d["steps"], d["redos"], r, fit, d["rate_w"], last["rms_w"], last["rms_du"],
            f(dr["S_rho"]), f(dr["S_m2"]), f(dr["S_m3"]), f(dr["S_E"]), f(dr["S_Etot"])))

md += ["", "Bitwise comparison base vs fix (sha256 of the final conserved state array and of the "
       "full-precision diagnostic CSV):", "",
       "| arm | final state identical | csv identical | max abs diff final state | state sha256 (base) |",
       "|---|---|---|---|---|"]
for n in names:
    if (n, "base") in res and (n, "fix") in res:
        a = np.load(os.path.join(R, "base", n + ".final_u.npy"))
        b = np.load(os.path.join(R, "fix", n + ".final_u.npy"))
        md.append("| %s | %s | %s | %.3g | %s |" % (
            n, res[(n, "base")]["state_sha256"] == res[(n, "fix")]["state_sha256"],
            res[(n, "base")]["csv_sha"] == res[(n, "fix")]["csv_sha"],
            float(np.abs(a - b).max()), res[(n, "base")]["state_sha256"][:16]))
open(OUT_MD, "w").write("\n".join(md) + "\n")
print("\n".join(md))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
col = {"base": "C3", "fix": "C0"}
for n in names:
    if "u400_s1" not in n:
        continue
    for tree in ("base", "fix"):
        d = res.get((n, tree))
        if d is None:
            continue
        t = np.array([r["t"] for r in d["rows"]]) / DAY
        ke = np.array([r["KE_pert"] for r in d["rows"]])
        lw = 2.0 if "dx100_" in n else 0.9
        ls = {"dx50": ":", "dx100": "-", "dx200": "--", "dx400": "-."}[n.split("_")[1]]
        ax[0].semilogy(t, np.array([r["rms_w"] for r in d["rows"]]), ls, color=col[tree], lw=lw,
                       label="%s %s (%s)" % (tree, n.split("_")[1],
                       "%+.1f/d" % d["rate"] if d["status"] == "finished" and "band" in d["fit"] and tree == "base"
                       else ("blew up" if d["status"] != "finished" else "%+.2f/d 0.5-3 d" % d["rate_w"])))
        ax[1].semilogy(t, np.maximum(ke, 1e-30), ls, color=col[tree], lw=lw,
                       label="%s %s" % (tree, n.split("_")[1]))
ax[0].set(xlabel="t (day)", ylabel="rms w (m/s)", title="scheme 1, u0 = 400 m/s: rms w")
ax[1].set(xlabel="t (day)", ylabel="perturbation KE (J)", title="0.5 rho (|u-u0|^2 + w^2), domain sum")
ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
for tree, mk in (("base", "o"), ("fix", "s")):
    d = res.get(("box_dx100_u400_s1", tree))
    if d is None:
        continue
    t = np.array([r["t"] for r in d["rows"]]) / DAY
    for key, c in (("S_rho", "k"), ("S_m2", "C2"), ("S_E", "C1"), ("S_Etot", "C4")):
        s = np.array([r[key] for r in d["rows"]])
        ax[2].semilogy(t, np.maximum(np.abs((s - s[0]) / s[0]), 1e-17), color=c,
                       ls="-" if tree == "fix" else ":", lw=1.5 if tree == "fix" else 1.0,
                       label="%s %s" % (tree, key[2:]))
ax[2].axhline(np.finfo(float).eps, color="gray", lw=0.6)
ax[2].set(xlabel="t (day)", ylabel="|S(t)-S(0)| / |S(0)|", title="dx 100 km box: global drift (m3 = 0 exactly)")
ax[2].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
