#!/usr/bin/env python3
"""hevi-ark2 on stratified columns whose top cells approach the density floor: the near-floor
Limit (Limits, docs/user_guide/configuration.rst) and the operator transient behind it.

coarse  45 levels over 40 scale heights (dz 106 km), isothermal 786 K, the H2-He-like gas of
        test_implicit_face_work_operator.py, gravity-work cell, reflecting walls,
        implicit-scheme 9 under rk3, dt 1500 s (vertical acoustic Courant 27.8), 40 steps.
        Its top density, about 1.5e-17 kg/m^3, is next to the 3.5e-18 floor, where the stage
        iteration does not contract, so any fixed count is a truncated iterate and the
        default's 5 solves retry (step 12). Pinned:
          * the rk path (implicit-integrator: rk) runs all NSTEP steps, finite,
            max|w| < W_BOUND;
          * the default (5 solves) either runs all NSTEP steps bounded, with a max relative
            error of rho and of E against a dt/16 reference (implicit-integrator-iterations
            20, run only in that case) at most RATIO times rk's, or its retry restores the
            step input exactly (the driver then halves dt).
        Fixed 4 and fixed 8 are reported, not asserted (the count is not monotone).
short   12 levels over 10.67 scale heights (the 106 km levels of the coarse column), gravity-
        work face, balance_column at its defaults (residual ~1e-10). A balanced column whose
        density falls over H scale heights amplifies a small balance residual by about
        e^{H/2} (max_t ||exp(tJ)|| of the explicit tendency's Jacobian); a consistent
        integrator reproduces that transient as dt -> 0, so converged hevi-ark2
        (implicit-integrator-iterations 20) at DT0/16 and DT0/64 over T_END: the peaks of
        max|w| agree within DT_RATIO, and max|w| / (c_s residual e^{H/2}) <= BOUND.

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
NZ_T, HEIGHTS_T = 12, 12 * 40 / 45
DT0, T_END = 1500.0, 6000.0
DT_RATIO, BOUND = 2.0, 0.3  # measured 1.115 and 0.113; BOUND 2.7x the measured peak


def coarse_run(keys, dt, nstep):
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
    wmax, done, status, rolled_back = 0.0, 0, "ok", None
    for n in range(nstep):
        previous = v["hydro_u"][sl].clone()
        for s in range(len(b.module("intg").stages)):
            b.forward(v, dt, s)
        if b.check_redo(v) != 0:
            status = "retry at step %d" % (n + 1)
            rolled_back = bool(torch.equal(v["hydro_u"][sl], previous))
            break
        u = v["hydro_u"][sl]
        if not torch.isfinite(u).all():
            status = "non-finite at step %d" % (n + 1)
            break
        wmax = max(wmax, (u[1] / u[kIDN]).abs().max().item())
        done = n + 1
    return dict(u=v["hydro_u"][sl].clone(), wmax=wmax, steps=done, status=status,
                rolled_back=rolled_back)


def errors(a, r):
    from snapy import kIDN, kIPR
    rel = lambda k: ((a[k] - r[k]) / r[k]).abs().max().item()
    return rel(kIDN), rel(kIPR)


def coarse(fails):
    fd = coarse_run({}, DT, NSTEP)
    rk = coarse_run({"implicit-integrator": "rk"}, DT, NSTEP)
    for name, r in (("default (5 solves)", fd), ("rk", rk)):
        print("%-18s %s, steps %d, max|w| %.3e, retry restored input: %s"
              % (name, r["status"], r["steps"], r["wmax"], r["rolled_back"]))
    if not (rk["status"] == "ok" and rk["steps"] == NSTEP and rk["wmax"] < W_BOUND):
        fails.append("rk not bounded: %s, steps %d, max|w| %.3e" % (rk["status"], rk["steps"], rk["wmax"]))
    if fd["status"] == "ok":
        if not (fd["steps"] == NSTEP and fd["wmax"] < W_BOUND):
            fails.append("default not bounded: steps %d, max|w| %.3e" % (fd["steps"], fd["wmax"]))
        elif rk["status"] == "ok":
            # the reference is needed only here: the default ran all steps
            ref = coarse_run({"implicit-integrator-iterations": 20}, DT / DTDIV, NSTEP * DTDIV)
            print("reference (dt/%d, iterations 20): %s" % (DTDIV, ref["status"]))
            if ref["status"] != "ok":
                fails.append("reference did not run: %s" % ref["status"])
            else:
                ed, er = errors(fd["u"], ref["u"]), errors(rk["u"], ref["u"])
                print("max relative error vs reference: default rho %.2e E %.2e; "
                      "rk rho %.2e E %.2e" % (ed[0], ed[1], er[0], er[1]))
                for k, name in ((0, "rho"), (1, "E")):
                    if not ed[k] <= RATIO * er[k]:
                        fails.append("default %s error %.2e > %.0f x rk's %.2e"
                                     % (name, ed[k], RATIO, er[k]))
    elif fd["rolled_back"] is not True:
        fails.append("default %s without restoring the step input" % fd["status"])
    # the Limit: other counts, reported only (the count is not monotone)
    for name, keys in (("fixed 4", {"implicit-integrator-iterations": 4}),
                       ("fixed 8", {"implicit-integrator-iterations": 8})):
        r = coarse_run(keys, DT, NSTEP)
        print("Limit, %-7s: %s, steps %d, max|w| %.3e" % (name, r["status"], r["steps"], r["wmax"]))


def short_run(dt, nstep, iterations=20):
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR

    H = fw.RD * fw.T0 / fw.GRAV
    dz = H * HEIGHTS_T / NZ_T
    cfg = fw.config(9)
    cfg["integration"].update({"implicit-integrator": "hevi-ark2",
                               "implicit-integrator-iterations": iterations})
    cfg["geometry"]["cells"]["nx1"] = NZ_T
    cfg["geometry"]["bounds"]["x1max"] = dz * NZ_T
    cfg["forcing"]["const-gravity"].update({"gravity-work": "face", "gravity-work-fixer": False})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "deck.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(cfg, f)
        b = MeshBlock(MeshBlockOptions.from_yaml(p))
    b.to(torch.device("cpu"), torch.float64)
    ng = 3
    z = (torch.arange(NZ_T, dtype=torch.float64) + 0.5) * dz
    pz = fw.PS * torch.exp(-z / H)
    col = torch.zeros(kIPR + 1, 1, 1, NZ_T, dtype=torch.float64)
    col[kIDN, 0, 0], col[kIPR, 0, 0] = pz / (fw.RD * fw.T0), pz
    wb, resid, _ = snapy.balance_column(col, torch.full((NZ_T,), dz, dtype=torch.float64), fw.GRAV)
    w = dict(b.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ_T] = wb[c, 0, 0]
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ_T:] = w[c][..., ng + NZ_T - 1:ng + NZ_T]
    v, _ = b.initialize({"hydro_w": w})
    sl = (Ellipsis, slice(ng, ng + fw.NX2), slice(ng, ng + NZ_T))
    peak, status = 0.0, "ok"
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
        peak = max(peak, (u[1] / u[kIDN]).abs().max().item())
    return peak, float(resid), status


def short(fails):
    cs = math.sqrt(fw.GAMMA * fw.RD * fw.T0)
    out = {}
    for div in (16, 64):
        dt = DT0 / div
        out[div] = short_run(dt, round(T_END / dt))
        print("dt/%d: peak max|w| %.3e m/s, balance residual %.2e, %s"
              % (div, out[div][0], out[div][1], out[div][2]), flush=True)
        if out[div][2] != "ok":
            fails.append("short column dt/%d: %s" % (div, out[div][2]))
    if not fails:
        p16, p64, resid = out[16][0], out[64][0], out[16][1]
        ratio = max(p16, p64) / min(p16, p64)
        scaled = max(p16, p64) / (cs * resid * math.exp(HEIGHTS_T / 2))
        print("peak ratio dt/16 vs dt/64 %.3f (<= %.1f); peak / (c_s residual e^{H/2}) %.3e (<= %.2e)"
              % (ratio, DT_RATIO, scaled, BOUND), flush=True)
        if not ratio <= DT_RATIO:
            fails.append("dt/16 and dt/64 peaks differ by %.3f" % ratio)
        if not scaled <= BOUND:
            fails.append("peak %.3e exceeds the operator bound (%.3e > %.2e)" % (max(p16, p64), scaled, BOUND))


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    fails = []
    coarse(fails)
    short(fails)
    print("FAILED: " + "; ".join(fails) if fails else "PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
