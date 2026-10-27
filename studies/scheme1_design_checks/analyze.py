#!/usr/bin/env python3
"""Tables for the scheme-1 design checks. usage: analyze.py [labels...] > tables.md

Per run (runs/<label>/<arm>.csv/.json/.prof.csv):
  rate: slope of ln(rms w) (1/day), rows 1e-4 <= rms w <= 1e-1 if >= 3, else t >= 0.5 d
  E+PE drift: (S_Etot(t) - S_Etot(0)) / S_Etot(0) at the end [max |.| over the run]
Scheme 1 vs 9 gaps (same label, same arm with s1/s9):
  rms w: max over rows t >= 600 s of |w1 - w9| / w9, and at 1, 2, 3 d
  u profile: max_k |<u>_1 - <u>_9| / u0 at 1, 2, 3 d (horizontal means)
  m_t profile: max_k |<m2>_1 - <m2>_9| / max_k |<m2>_9|
"""
import csv, json, math, os, sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
DAY = 86400.0


def load(label, arm):
    b = os.path.join(here, "runs", label, arm)
    if not os.path.exists(b + ".csv"):
        return None
    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(b + ".csv"))]
    js = json.load(open(b + ".json")) if os.path.exists(b + ".json") else None
    prof = {}
    if os.path.exists(b + ".prof.csv"):
        for r in csv.DictReader(open(b + ".prof.csv")):
            prof.setdefault(round(float(r["t"]) / DAY, 1), []).append(
                (float(r["m2_mean"]), float(r["u_mean"]), float(r["u_init"])))
    return {"rows": rows, "js": js, "prof": {k: np.array(v) for k, v in prof.items()}}


def rate(rows):
    t = np.array([r["t"] for r in rows]) / DAY
    w = np.array([r["rms_w"] for r in rows])
    ok = np.isfinite(w) & (w > 0)
    band = ok & (w >= 1e-4) & (w <= 1e-1)
    sel = band if band.sum() >= 3 else ok & (t >= 0.5)
    if sel.sum() < 3:
        return float("nan"), "none"
    return np.polyfit(t[sel], np.log(w[sel]), 1)[0], ("band" if band.sum() >= 3 else "run")


def at(rows, d, k):
    r = min(rows, key=lambda r: abs(r["t"] - d * DAY))
    return r[k]


def etot(rows):
    e0 = rows[0]["S_Etot"]
    d = [(r["S_Etot"] - e0) / abs(e0) for r in rows]
    return d[-1], max(abs(x) for x in d)


def stopped(label, arm):
    st = os.path.join(here, "runs", label, arm + ".state")
    txt = open(st).read() if os.path.exists(st) else ""
    return "stopped (" + txt.split("exit ")[1].split()[0] + ")" if "exit " in txt else "running"


def g(x, f="%.2e"):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f % x


labels = sys.argv[1:] or ["base", "fix", "roevt"]
arms = ["box_dx50_u400", "box_dx100_u400", "box_dx200_u400", "shearbox_dx100_u400", "col_u400", "coln2_u400", "shear_u400", "shearn2_u400", "moist_u400"]
out = ["## Per run\n",
       "| label | arm | status | t_end (d) | redos | dt median (s) | rate (1/d) | fit | rms w @1/2/3 d | max\\|u-u_init\\|/u0 over run | E+PE drift end [max] |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
for lab in labels:
    for a in arms:
        for s in ("1", "9", "0"):
            r = load(lab, f"{a}_s{s}")
            if r is None:
                continue
            rows, js = r["rows"], r["js"]
            rt, how = rate(rows)
            e, em = etot(rows)
            mx = max(x["max_du_rel"] for x in rows[1:])
            out.append("| %s | %s_s%s | %s | %.3f | %s | %s | %s | %s | %s / %s / %s | %s | %s [%s] |" % (
                lab, a, s, js["status"] if js else stopped(lab, a + "_s" + s), rows[-1]["t"] / DAY, js["redos"] if js else "n/a",
                g(js and js["dt_median"], "%.2f"), g(rt, "%.3f"), how,
                g(at(rows, 1, "rms_w")), g(at(rows, 2, "rms_w")), g(at(rows, 3, "rms_w")), g(mx), g(e), g(em)))

out += ["\n## Scheme 1 vs scheme 9 (same label, same arm)\n",
        "| label | arm | rms w gap max over run | rms w gap @1/2/3 d | u-profile gap /u0 @1/2/3 d | m_t-profile gap @1/2/3 d | max\\|<u>-u_init\\|/u0 @3 d s1 / s9 |",
        "|---|---|---|---|---|---|---|"]
for lab in labels:
    for a in arms:
        r1, r9 = load(lab, a + "_s1"), load(lab, a + "_s9")
        if not r1 or not r9 or not r1["js"] or not r9["js"]:
            continue
        u0 = r1["js"]["u0"]
        n = min(len(r1["rows"]), len(r9["rows"]))
        gaps = [abs(r1["rows"][k]["rms_w"] - r9["rows"][k]["rms_w"]) / r9["rows"][k]["rms_w"]
                for k in range(n) if r1["rows"][k]["t"] >= 600 and r9["rows"][k]["rms_w"] > 0]
        gd = [abs(at(r1["rows"], d, "rms_w") - at(r9["rows"], d, "rms_w")) / at(r9["rows"], d, "rms_w") for d in (1, 2, 3)]
        up, mp, dev = [], [], []
        for d in (1.0, 2.0, 3.0):
            p1, p9 = r1["prof"].get(d), r9["prof"].get(d)
            if p1 is None or p9 is None:
                up.append(float("nan")); mp.append(float("nan")); continue
            up.append(np.abs(p1[:, 1] - p9[:, 1]).max() / u0)
            mp.append(np.abs(p1[:, 0] - p9[:, 0]).max() / np.abs(p9[:, 0]).max())
        for p in (r1["prof"].get(3.0), r9["prof"].get(3.0)):
            dev.append(np.abs(p[:, 1] - p[:, 2]).max() / u0 if p is not None else float("nan"))
        out.append("| %s | %s | %s | %s / %s / %s | %s / %s / %s | %s / %s / %s | %s / %s |" % (
            lab, a, g(max(gaps) if gaps else float("nan")), *[g(x) for x in gd], *[g(x) for x in up],
            *[g(x) for x in mp], *[g(x) for x in dev]))
print("\n".join(out))
