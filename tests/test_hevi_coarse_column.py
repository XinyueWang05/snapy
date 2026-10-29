#!/usr/bin/env python3
"""hevi-ark2 on a deep stratified column whose top cells sit near the density floor.

Test case for the open item "why the hevi-ark2 stage iteration fails to contract near the
density floor" (Limit, docs/user_guide/configuration.rst). The column: 45 levels
over 40 scale heights (dz 106 km), isothermal 786 K, H2-He-like gas of
test_implicit_face_work_operator.py, gravity-work cell, reflecting walls, implicit-scheme 9
under rk3 (hevi-ark2 by default), dt 1500 s (vertical acoustic Courant 27.8), 40 steps.
Its top density, about 1.5e-17 kg/m^3, is next to the 3.5e-18 floor. There the stage
iteration does not contract (measured: no stage converges in 60 solves; fixed 8 fails at
step 14), so the default's 4 solves are a truncated iterate. This test pins the current
behaviour:

  * the default (4 solves) runs all NSTEP steps, finite, without a retry, max|w| < W_BOUND;
  * its error against a dt/16 reference at the same time (hevi-ark2 with
    implicit-integrator-tolerance 1e-10, which converges at dt/16 in at most 5 solves) is
    comparable to the rk path's (implicit-integrator: rk) at the same dt: max relative
    error of rho and of E at most RATIO times rk's (measured 2.2e-5 vs 2.1e-5).

Fixed 8 and the stop-on-no-contraction rule are run and reported, not asserted (they fail
on this column today; resolving the open item should let them through).

  python test_hevi_coarse_column.py
"""
import math
import os
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_implicit_face_work_operator as fw  # noqa: E402

NZ, HEIGHTS, DT, NSTEP, DTDIV = 45, 40, 1500.0, 40, 16
W_BOUND, RATIO = 1.0, 2.0


def run(keys, dt, nstep):
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR

    H = fw.RD * fw.T0 / fw.GRAV
    dz = H * HEIGHTS / NZ
    cfg = fw.config(9)
    cfg["integration"].update(keys)
    cfg["geometry"]["cells"]["nx1"] = NZ
    cfg["geometry"]["bounds"]["x1max"] = dz * NZ
    cfg["forcing"]["const-gravity"].update({"gravity-work": "cell", "gravity-work-fixer": False})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "deck.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(cfg, f)
        b = MeshBlock(MeshBlockOptions.from_yaml(p))
    b.to(torch.device("cpu"), torch.float64)
    ng = 3
    z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * dz
    pz = fw.PS * torch.exp(-z / H)
    col = torch.zeros(kIPR + 1, 1, 1, NZ, dtype=torch.float64)
    col[kIDN, 0, 0], col[kIPR, 0, 0] = pz / (fw.RD * fw.T0), pz
    wb, _, _ = snapy.balance_column(col, torch.full((NZ,), dz, dtype=torch.float64), fw.GRAV)
    w = dict(b.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ] = wb[c, 0, 0]
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ:] = w[c][..., ng + NZ - 1:ng + NZ]
    v, _ = b.initialize({"hydro_w": w})
    sl = (Ellipsis, slice(ng, ng + fw.NX2), slice(ng, ng + NZ))
    wmax, done, status = 0.0, 0, "ok"
    for n in range(nstep):
        for s in range(len(b.module("intg").stages)):
            b.forward(v, dt, s)
        if b.check_redo(v) != 0:
            status = "retry at step %d" % (n + 1)
            break
        u = v["hydro_u"][sl]
        if not torch.isfinite(u).all():
            status = "non-finite at step %d" % (n + 1)
            break
        wmax = max(wmax, (u[1] / u[kIDN]).abs().max().item())
        done = n + 1
    return dict(u=v["hydro_u"][sl].clone(), wmax=wmax, steps=done, status=status,
                stats=b.hevi_solve_stats().tolist())


def errors(a, r):
    from snapy import kIDN, kIPR
    rel = lambda k: ((a[k] - r[k]) / r[k]).abs().max().item()
    return rel(kIDN), rel(kIPR)


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    fails = []
    ref = run({"implicit-integrator-tolerance": 1.e-10}, DT / DTDIV, NSTEP * DTDIV)
    f4 = run({}, DT, NSTEP)
    rk = run({"implicit-integrator": "rk"}, DT, NSTEP)
    print("reference (dt/%d, tolerance 1e-10): %s, solve stats %s" % (DTDIV, ref["status"], ref["stats"]))
    for name, r in (("default (4 solves)", f4), ("rk", rk)):
        print("%-18s %s, steps %d, max|w| %.3e" % (name, r["status"], r["steps"], r["wmax"]))
    if ref["status"] != "ok" or ref["stats"][3] > 0:
        fails.append("reference did not run or did not converge: %s %s" % (ref["status"], ref["stats"]))
    if not (f4["status"] == "ok" and f4["steps"] == NSTEP and f4["wmax"] < W_BOUND):
        fails.append("default not bounded: %s, steps %d, max|w| %.3e" % (f4["status"], f4["steps"], f4["wmax"]))
    if rk["status"] == "ok" and ref["status"] == "ok" and f4["status"] == "ok":
        e4, er = errors(f4["u"], ref["u"]), errors(rk["u"], ref["u"])
        print("max relative error vs reference: default rho %.2e E %.2e; rk rho %.2e E %.2e"
              % (e4[0], e4[1], er[0], er[1]))
        for k, name in ((0, "rho"), (1, "E")):
            if not e4[k] <= RATIO * er[k]:
                fails.append("default %s error %.2e > %.0f x rk's %.2e" % (name, e4[k], RATIO, er[k]))
    else:
        fails.append("rk or the default did not complete")
    # open item: more solves diverge today (reported only)
    for name, keys in (("fixed 8", {"implicit-integrator-iterations": 8,
                                    "implicit-integrator-tolerance": 0.0}),
                       ("stop rule", {"implicit-integrator-tolerance": 1.e-10})):
        r = run(keys, DT, NSTEP)
        print("open item, %-9s: %s, steps %d, solve stats %s" % (name, r["status"], r["steps"], r["stats"]))
    print("FAILED: " + "; ".join(fails) if fails else "PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
