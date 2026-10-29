#!/usr/bin/env python3
"""Converged hevi-ark2 follows the operator's own transient on a short stratified column.

A balanced column whose density falls over HEIGHTS scale heights amplifies a small balance
residual on its way up: the operator itself (the explicit tendency's Jacobian at the rest
state) has max_t ||exp(tJ)|| ~ e^{H/2} (measured on this deck family: 1.6e2, 2.1e4, 1.5e6,
3.3e8 at 10.7, 20.4, 30.2, 40 H; e^{H/2} = 2.1e2, 2.8e4, 3.7e6, 4.9e8). A consistent integrator
reproduces that transient as dt -> 0, so

  * converged hevi-ark2 (implicit-integrator-iterations 20, no tolerance key) at dt/16 and
    dt/64 agrees: the two peaks of max|w| over the run are within a factor DT_RATIO;
  * the peak stays within the operator's bound: max|w| / (c_s * residual * e^{H/2}) <= BOUND,
    with residual the balance_column residual of the initial column.

Column: 12 levels over 10.67 scale heights (the 106 km levels of the 45-level, 40 H coarse
column), isothermal 786 K, gas and walls of test_implicit_face_work_operator.py, gravity-work
face, implicit-scheme 9 under rk3, balance_column at its defaults (residual ~1e-10), T_END s.

  python test_hevi_short_column_transient.py
"""
import math
import os
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_implicit_face_work_operator as fw  # noqa: E402

NZ, HEIGHTS = 12, 12 * 40 / 45
DT0, T_END = 1500.0, 6000.0
DT_RATIO, BOUND = 2.0, 0.3  # measured 1.115 and 0.113; BOUND 2.7x the measured peak


def run(dt, nstep, iterations=20):
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR

    H = fw.RD * fw.T0 / fw.GRAV
    dz = H * HEIGHTS / NZ
    cfg = fw.config(9)
    cfg["integration"].update({"implicit-integrator": "hevi-ark2",
                               "implicit-integrator-iterations": iterations})
    cfg["geometry"]["cells"]["nx1"] = NZ
    cfg["geometry"]["bounds"]["x1max"] = dz * NZ
    cfg["forcing"]["const-gravity"].update({"gravity-work": "face", "gravity-work-fixer": False})
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
    wb, resid, _ = snapy.balance_column(col, torch.full((NZ,), dz, dtype=torch.float64), fw.GRAV)
    w = dict(b.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ] = wb[c, 0, 0]
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ:] = w[c][..., ng + NZ - 1:ng + NZ]
    v, _ = b.initialize({"hydro_w": w})
    sl = (Ellipsis, slice(ng, ng + fw.NX2), slice(ng, ng + NZ))
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


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    cs = math.sqrt(fw.GAMMA * fw.RD * fw.T0)
    fails, out = [], {}
    for div in (16, 64):
        dt = DT0 / div
        out[div] = run(dt, round(T_END / dt))
        print("dt/%d: peak max|w| %.3e m/s, balance residual %.2e, %s"
              % (div, out[div][0], out[div][1], out[div][2]), flush=True)
        if out[div][2] != "ok":
            fails.append("dt/%d: %s" % (div, out[div][2]))
    if not fails:
        p16, p64, resid = out[16][0], out[64][0], out[16][1]
        ratio = max(p16, p64) / min(p16, p64)
        scaled = max(p16, p64) / (cs * resid * math.exp(HEIGHTS / 2))
        print("peak ratio dt/16 vs dt/64 %.3f (<= %.1f); peak / (c_s residual e^{H/2}) %.3e (<= %.2e)"
              % (ratio, DT_RATIO, scaled, BOUND), flush=True)
        if not ratio <= DT_RATIO:
            fails.append("dt/16 and dt/64 peaks differ by %.3f" % ratio)
        if not scaled <= BOUND:
            fails.append("peak %.3e exceeds the operator bound (%.3e > %.2e)" % (max(p16, p64), scaled, BOUND))
    print("FAILED: " + "; ".join(fails) if fails else "PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
