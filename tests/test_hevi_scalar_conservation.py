#!/usr/bin/env python3
"""Passive scalars through the hevi-ark2 (ARS(2,3,2)) step.

A dry isothermal column (x1 vertical, 32 levels of 1 km, reflecting walls; x2 periodic,
8 cells of 2 km) in the solver's discrete balance, with a vertical wave w = 5 sin(pi z / H),
a sheared wind u = 10 + 5 cos(pi z / H) and a 0.5 K thermal blob, carries two tracers:
a uniform ratio R0 and a blob ratio R0 (1 + exp(-((x - x0)/L)^2 - ((z - z0)/H0)^2)).
implicit-scheme 9 under rk3 with no implicit-integrator key runs hevi-ark2. After NSTEP steps:

  * the step is hevi-ark2 (its solve counter moved);
  * total tracer mass sum(s V) of each tracer: relative change <= TOL;
  * the uniform tracer stays uniform: max |s / (rho R0) - 1| <= TOL;
  * the blob tracer moved (the test is not vacuous).

  python test_hevi_scalar_conservation.py
"""
import math
import os
import sys
import tempfile

import torch
import yaml

G, T0, PS, RD = 10.0, 300.0, 1.0e5, 287.0
NZ, DZ, NX2, DX = 32, 1.0e3, 8, 2.0e3
NSTEP, R0, TOL = 40, 1.0e-3, 1.0e-12


def config():
    return {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry", "composition": {"N": 1.56, "O": 0.42, "Ar": 0.01},
                     "cv_R": 2.5}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": NZ * DZ, "x2min": 0., "x2max": NX2 * DX,
                                "x3min": 0., "x3max": DX},
                     "cells": {"nx1": NZ, "nx2": NX2, "nx3": 1, "nghost": 3}},
        "dynamics": {"equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                           "pressure-floor": 1.e-10, "limiter": False},
                     "reconstruct": {"vertical": {"type": "weno5", "scale": False, "shock": False},
                                     "horizontal": {"type": "weno5", "scale": False,
                                                    "shock": False}},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.9, "implicit-scheme": 9, "nlim": -1,
                        "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -G}},
        "scalar": {"nvar": 2, "names": ["uniform", "blob"],
                   "reconstruct": {"type": "weno5", "scale": False, "shock": False}},
    }


def main():
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

    torch.set_default_dtype(torch.float64)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "deck.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(config(), f)
        block = MeshBlock(MeshBlockOptions.from_yaml(p))
    block.to(torch.device("cpu"))

    ng = 3
    z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * DZ
    x = (torch.arange(NX2, dtype=torch.float64) + 0.5) * DX
    pz = PS * torch.exp(-G * z / (RD * T0))
    col = torch.zeros(kIPR + 1, 1, 1, NZ, dtype=torch.float64)
    col[kIDN, 0, 0], col[kIPR, 0, 0] = pz / (RD * T0), pz
    wb, _, _ = snapy.balance_column(col, torch.full((NZ,), DZ, dtype=torch.float64), G,
                                    rtol=1.e-12, max_iter=400)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ] = wb[c, 0, 0]
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ:] = w[c][..., ng + NZ - 1:ng + NZ]
    H = NZ * DZ
    zz = torch.zeros(w.shape[-1])
    zz[ng:ng + NZ] = z
    xx = torch.zeros(w.shape[-2])
    xx[ng:ng + NX2] = x
    blob = torch.exp(-((xx.view(-1, 1) - 0.3 * NX2 * DX) / (2 * DX)) ** 2
                     - ((zz.view(1, -1) - 0.4 * H) / (0.15 * H)) ** 2)
    w[kIDN] = w[kIDN] / (1.0 + 0.5 / T0 * blob)  # +0.5 K at fixed p
    w[kIV1] = 5.0 * torch.sin(math.pi * zz / H).view(1, 1, -1)
    w[kIV1 + 1] = (10.0 + 5.0 * torch.cos(math.pi * zz / H)).view(1, 1, -1)
    r = torch.full((2,) + tuple(w.shape[1:]), R0, dtype=torch.float64)
    r[1] = R0 * (1.0 + blob)
    v, _ = block.initialize({"hydro_w": w, "scalar_r": r})

    I = (slice(None), 0, slice(ng, ng + NX2), slice(ng, ng + NZ))
    vol = DX * DX * DZ  # uniform Cartesian cells
    mass0 = (v["scalar_s"][I] * vol).sum(dim=(-1, -2))
    blob0 = v["scalar_s"][1][I[1:]].clone()
    stats0 = block.hevi_solve_stats().clone()
    for _ in range(NSTEP):
        dt = block.max_time_step(v)
        for s in range(len(block.module("intg").stages)):
            block.forward(v, dt, s)
        assert block.check_redo(v) == 0
    stats = block.hevi_solve_stats()
    mass = (v["scalar_s"][I] * vol).sum(dim=(-1, -2))
    rel = ((mass - mass0) / mass0).abs()
    unif = (v["scalar_s"][0][I[1:]] / (v["hydro_u"][kIDN][I[1:]] * R0) - 1.0).abs().max().item()
    moved = (v["scalar_s"][1][I[1:]] - blob0).abs().max().item() / blob0.abs().max().item()
    print("hevi-ark2 solve stats (stages, solves, max per stage, capped): %s"
          % (stats - stats0).tolist())
    print("tracer mass relative change: uniform %.3e, blob %.3e (tol %.0e)"
          % (rel[0].item(), rel[1].item(), TOL))
    print("uniform tracer max |s/(rho R0) - 1| = %.3e; blob moved by %.3e of its max"
          % (unif, moved))
    fails = []
    if not (stats - stats0)[0].item() > 0:
        fails.append("hevi-ark2 did not run (solve counter did not move)")
    if not rel.max().item() <= TOL:
        fails.append("tracer mass not conserved: %.3e" % rel.max().item())
    if not unif <= TOL:
        fails.append("uniform tracer departed: %.3e" % unif)
    if not moved > 1.e-3:
        fails.append("blob did not move (%.3e)" % moved)
    print("FAILED: " + "; ".join(fails) if fails else "PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
