#!/usr/bin/env python3
"""implicit-integrator hevi-ark2 at its default stage iterations (key absent) and at 5.

The u = 0 box without noise at vertical acoustic Courant 6.7: x1 vertical, 100 levels of
4.5 km (about 10 scale heights), reflecting walls; x2 periodic, 4 cells of 100 km; dry
H2-He, isothermal 100 K, 10 bar at the bottom, balanced with snapy.balance_column to rtol
1e-12 (400 sweeps), so the bound measures the solver and not the IC residual. Fixed
dt, NSTEP steps; w starts at round-off. One linearised solve per stage (iterations 1)
grows here: max|w| rises about 100x to above 1e-5 in 480 steps. Each arm:

  * finite, and max|w| over the run stays below W_BOUND;
  * decays: the largest max|w| of the last quarter of the run is below DECAY times the
    largest max|w| of the first quarter;
  * mass: the relative change of the total mass is below MASS_TOL.

Also: the default arm is bitwise the arm with implicit-integrator-iterations 4, and the
arm at 5 is not bitwise the arm at 4 (the key reaches the solver).

  python test_hevi_iterations.py
"""
import math
import os
import sys
import tempfile

import torch
import yaml

G, T0, PS = 8.87, 100.0, 1.0e6
NZ, DZ, NX2, DX = 100, 4.5e3, 4, 100.e3
COURANT, NSTEP = 6.7, 480
# W_BOUND: 7.8x the measured peak (1.288e-9 default, 1.286e-9 at 5) from the column
# balanced to rtol 1e-12 (residual 6.5e-13); the default-tolerance column (5.8e-11)
# peaked at 1.26e-7
W_BOUND, DECAY, MASS_TOL = 1.e-8, 0.5, 1.e-13


def config(iterations):
    return {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry", "composition": {"H": 1.5, "He": 0.15}, "cv_R": 2.5}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": NZ * DZ, "x2min": 0., "x2max": NX2 * DX,
                                "x3min": 0., "x3max": DX},
                     "cells": {"nx1": NZ, "nx2": NX2, "nx3": 1, "nghost": 3}},
        "dynamics": {"equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                           "pressure-floor": 1.e-10, "limiter": True,
                                           "max-iter": 10, "ftol": 1.e-6},
                     "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": True},
                                     "horizontal": {"type": "weno5", "scale": True, "shock": True}},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.3, "implicit-scheme": 9, "nlim": -1, "tlim": 1.e9,
                        "implicit-integrator": "hevi-ark2",
                        **({} if iterations is None else {"implicit-integrator-iterations": iterations})},
        "forcing": {"const-gravity": {"grav1": -G}},
    }


def run(iterations, tmp):
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1
    p = os.path.join(tmp, "deck.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(config(iterations), f)
    block = MeshBlock(MeshBlockOptions.from_yaml(p))
    block.to(torch.device("cpu"))

    ng = 3
    rd = 8.31446261815324 / (1.5 * 1.00794e-3 + 0.15 * 4.002602e-3)
    z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * DZ
    pz = PS * torch.exp(-G * z / (rd * T0))
    col = torch.zeros(kIPR + 1, 1, 1, NZ, dtype=torch.float64)
    col[kIDN, 0, 0], col[kIPR, 0, 0] = pz / (rd * T0), pz
    wb, resid, _ = snapy.balance_column(col, torch.full((NZ,), DZ, dtype=torch.float64), G,
                                        rtol=1.e-12, max_iter=400)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ] = wb[c, 0, 0]
    for c in (kIDN, kIPR):
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ:] = w[c][..., ng + NZ - 1:ng + NZ]
    bv, _ = block.initialize({"hydro_w": w})

    inner = (slice(None), 0, slice(ng, ng + NX2), slice(ng, ng + NZ))
    dt = COURANT * DZ / math.sqrt(1.4 * rd * T0)  # cv_R 2.5: gamma 1.4
    mass0 = bv["hydro_u"][inner][kIDN].sum().item()
    wmax = []
    for _ in range(NSTEP):
        for s in range(len(block.module("intg").stages)):
            block.forward(bv, dt, s)
        u = bv["hydro_u"][inner]
        wmax.append((u[kIV1] / u[kIDN]).abs().max().item())
    u = bv["hydro_u"][inner]
    return dict(u=bv["hydro_u"].clone(), wmax=wmax, dt=dt, resid=float(resid),
                mass_rel=abs(u[kIDN].sum().item() - mass0) / mass0)


def check(name, r, fails):
    q = NSTEP // 4
    first, last = max(r["wmax"][:q]), max(r["wmax"][-q:])
    peak = max(r["wmax"])
    finite = bool(torch.isfinite(r["u"]).all()) and all(math.isfinite(x) for x in r["wmax"])
    print("%-12s dt %.3f s: max|w| peak %.3e, first quarter %.3e, last quarter %.3e "
          "(ratio %.3f), end %.3e; mass drift %.2e"
          % (name, r["dt"], peak, first, last, last / first, r["wmax"][-1], r["mass_rel"]))
    if not finite:
        fails.append("%s: not finite" % name)
    elif peak >= W_BOUND:
        fails.append("%s: max|w| %.3e >= %.1e" % (name, peak, W_BOUND))
    elif not last < DECAY * first:
        fails.append("%s: not decaying (last/first quarter %.3f >= %.2f)" % (name, last / first, DECAY))
    if not r["mass_rel"] < MASS_TOL:
        fails.append("%s: mass drift %.2e >= %.0e" % (name, r["mass_rel"], MASS_TOL))


def main():
    torch.set_default_dtype(torch.float64)
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        r_def, r4, r5 = run(None, tmp), run(4, tmp), run(5, tmp)
    print("initial column: balance_column residual %.2e (rtol 1e-12, 400 sweeps)" % r_def["resid"])
    check("default", r_def, fails)
    check("iterations 5", r5, fails)
    same4 = torch.equal(r_def["u"], r4["u"])
    diff5 = not torch.equal(r5["u"], r4["u"])
    print("default bitwise = iterations 4: %s; iterations 5 differs from 4: %s (max |u5 - u4| %.3e)"
          % (same4, diff5, (r5["u"] - r4["u"]).abs().max().item()))
    if not same4:
        fails.append("default is not bitwise iterations 4")
    if not diff5:
        fails.append("iterations 5 is bitwise iterations 4")
    if fails:
        print("FAILED:", "; ".join(fails))
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
