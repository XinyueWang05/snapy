#!/usr/bin/env python3
"""vic-full leaves the explicit horizontal flux increment out of the vertical solve.

A discretely balanced isothermal column (snapy.balance_column, reflecting x1 walls,
const gravity, rk3 + weno5 + lmars) carries a wind U0 + A sin(2 pi x / L) along x2,
uniform in height, with no density or pressure perturbation. Its explicit increment
is horizontal advection only; the vertical one is the round-off of the balance. One
rk3 stage with implicit-scheme 9:
  * the implicit correction (hydro.icorr.corr) stays at round-off relative to the
    explicit increment in rho, m2 and E (REL_TOL). Before the fix the solve filtered
    the horizontal increment through (I + dt J)^-1 and moved 2-4e-3 of it.

  python test_implicit_horizontal_bypass.py [--device cpu]
"""
import argparse
import os
import sys
import tempfile

import torch
import yaml

GRAV, T0, PS = 8.87, 100.0, 1.0e6
NZ, DZ, NX2, DX = 24, 4500.0, 16, 5000.0
U0, AMP = 400.0, 20.0
REL_TOL = 1.0e-8


def config(scheme):
    return {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry", "composition": {"H": 1.5, "He": 0.15}, "cv_R": 2.5}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": NZ * DZ, "x2min": 0.0, "x2max": NX2 * DX,
                                "x3min": 0.0, "x3max": DX},
                     "cells": {"nx1": NZ, "nx2": NX2, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                  "pressure-floor": 1.e-10, "limiter": True,
                                  "max-iter": 10, "ftol": 1.e-6},
            "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": True},
                            "horizontal": {"type": "weno5", "scale": True, "shock": True}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {
            "x1-inner": "reflecting", "x1-outer": "reflecting",
            "x2-inner": "periodic", "x2-outer": "periodic",
            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.3, "implicit-scheme": scheme, "nlim": -1,
                        "tlim": 1.0e6, "ncycle_out": 1000000},
        "forcing": {"const-gravity": {"grav1": -GRAV}},
    }


def run(snapy, scheme, u0, amp, device, tmp):
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV2
    path = os.path.join(tmp, "s%d.yaml" % scheme)
    with open(path, "w") as f:
        yaml.safe_dump(config(scheme), f, sort_keys=False)
    block = MeshBlock(MeshBlockOptions.from_yaml(path))
    block.to(torch.device(device))
    ng = 3
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    rd = 8.31446261815324 / (1.5 * 1.00794e-3 + 0.15 * 4.002602e-3)
    z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * DZ
    p = PS * torch.exp(-GRAV * z / (rd * T0))
    col = torch.zeros(kIPR + 1, 1, 1, NZ, dtype=torch.float64)
    col[kIDN, 0, 0] = p / (rd * T0)
    col[kIPR, 0, 0] = p
    wb, _, _ = snapy.balance_column(col, torch.full((NZ,), DZ, dtype=torch.float64), GRAV)
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ] = wb[c, 0, 0].to(w)
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ:] = w[c][..., ng + NZ - 1:ng + NZ]
    x = (torch.arange(w.shape[2], dtype=torch.float64) - ng + 0.5) * DX
    w[kIV2] = (u0 + amp * torch.sin(2 * torch.pi * x / (NX2 * DX))).view(1, -1, 1).to(w)
    bv, _ = block.initialize({"hydro_w": w})
    dt = 0.3 * DX / (u0 + amp + 742.0)
    u_init = bv["hydro_u"].clone()
    block.forward(bv, dt, 0)
    inner = (slice(None), 0, slice(ng, ng + NX2), slice(ng, ng + NZ))
    du = (bv["hydro_u"] - u_init)[inner]
    nb = dict(block.named_buffers())
    corr = nb["hydro.icorr.corr"][inner].clone() if scheme else None
    return du, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    import snapy
    from snapy import kIDN, kIPR, kIV2
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        du0, _ = run(snapy, 0, U0, AMP, args.device, tmp)
        du9, corr = run(snapy, 9, U0, AMP, args.device, tmp)
        for k, name in ((kIDN, "rho"), (kIV2, "m2"), (kIPR, "E")):
            rel = (corr[k].pow(2).mean().sqrt() / du0[k].pow(2).mean().sqrt()).item()
            ok = rel < REL_TOL
            print("wind  %-3s |corr| / |du explicit| = %.3e  (tol %.0e) %s" % (name, rel, REL_TOL, "ok" if ok else "FAIL"))
            if not ok:
                fails.append("wind " + name)
    if fails:
        print("FAILED:", ", ".join(fails))
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
