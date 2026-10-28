#!/usr/bin/env python3
"""Constant forcing through the hevi-ark2 (ARS(2,3,2)) step, against u(t) = a t.

A dry isothermal column at rest in the solver's discrete balance (x1 vertical, 40 levels
of 1 km, reflecting walls; x2 periodic, 4 cells) is pushed by a uniform horizontal
acceleration a. With rho fixed and the flow uniform in x2, the exact answer is
u(t) = a t, w = 0. Two arms, each on hevi-ark2 (implicit-scheme 9, rk3, no
implicit-integrator key):

  native  const-gravity grav2 = a: a native forcing, in the vertical part V
  user    a TorchScript user stage forcing hydro_du[IVY] = dt rho a with its work
          hydro_du[IPR] = dt a rho u: explicit, in H

Each arm: max |u - a t| / (a t) <= TOL_U after NSTEP steps of fixed dt, max |w| <= W_BOUND,
and the solve counter moved (the step is hevi-ark2). The rk path (implicit-integrator: rk)
is reported for reference (it meets u = a t as well).

  python test_hevi_constant_forcing.py
"""
import math
import os
import sys
import tempfile

import torch
import yaml

G, T0, PS, RD = 10.0, 300.0, 1.0e5, 287.0
NZ, DZ, NX2, DX = 40, 1.0e3, 4, 10.0e3
A, NSTEP, DT = 1.0e-3, 50, 20.0
# W_BOUND: 3.5x the rk path's 2.9e-8 on the same box (the column's balance residual)
TOL_U, W_BOUND = 1.0e-10, 1.0e-7


class ConstantPush(torch.nn.Module):
    def __init__(self, a: float):
        super().__init__()
        self.a = a

    def forward(self, var: dict[str, torch.Tensor], dt: float, stage: int) -> dict[str, torch.Tensor]:
        u = var["hydro_u"]
        du = torch.zeros_like(u)
        du[2] = dt * u[0] * self.a
        du[4] = dt * u[2] * self.a  # the work a (rho u), as const-gravity grav2 books it
        return {"hydro_du": du}


def config(native, integrator=None):
    grav = {"grav1": -G}
    if native:
        grav["grav2"] = A
    return {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry", "composition": {"N": 1.56, "O": 0.42, "Ar": 0.01},
                     "cv_R": 2.5}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": NZ * DZ, "x2min": 0., "x2max": NX2 * DX,
                                "x3min": 0., "x3max": DX},
                     "cells": {"nx1": NZ, "nx2": NX2, "nx3": 1, "nghost": 3}},
        "dynamics": {"equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                           "pressure-floor": 1.e-10, "limiter": True},
                     "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": True},
                                     "horizontal": {"type": "weno5", "scale": True,
                                                    "shock": True}},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": dict({"type": "rk3", "cfl": 0.9, "implicit-scheme": 9, "nlim": -1,
                             "tlim": 1.e9},
                            **({"implicit-integrator": integrator} if integrator else {})),
        "forcing": {"const-gravity": grav},
    }


def run(native, tmp, integrator=None):
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

    p = os.path.join(tmp, "deck.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(config(native, integrator), f)
    block = MeshBlock(MeshBlockOptions.from_yaml(p))
    block.to(torch.device("cpu"))
    if not native:
        path = os.path.join(tmp, "push.pt")
        torch.jit.script(ConstantPush(A).eval()).save(path)
        block.set_user_stage_forcings([path])

    ng = 3
    z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * DZ
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
    v, _ = block.initialize({"hydro_w": w})
    stats0 = block.hevi_solve_stats().clone()
    for _ in range(NSTEP):
        for s in range(len(block.module("intg").stages)):
            block.forward(v, DT, s)
    I = (0, slice(ng, ng + NX2), slice(ng, ng + NZ))
    # velocities from the conserved state: hydro_w is refreshed at the start of each
    # stage's forward, so after a step it holds the last stage's input (t + dt/2 for
    # rk3's third stage, t + dt for ARS(2,3,2)), not u^{n+1}
    hu = v["hydro_u"]
    u = (hu[kIV1 + 1] / hu[kIDN])[I]
    wv = (hu[kIV1] / hu[kIDN])[I]
    exact = A * NSTEP * DT
    return dict(err=((u - exact).abs().max() / exact).item(), w=wv.abs().max().item(),
                solves=(block.hevi_solve_stats() - stats0).tolist())


def main():
    torch.set_default_dtype(torch.float64)
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, native in (("native grav2 (V)", True), ("user forcing (H)", False)):
            r = run(native, tmp)
            print("%-18s max|u - a t|/(a t) %.3e (tol %.0e), max|w| %.3e, solve stats %s"
                  % (name, r["err"], TOL_U, r["w"], r["solves"]))
            if not r["err"] <= TOL_U:
                fails.append("%s: u error %.3e" % (name, r["err"]))
            if not r["w"] <= W_BOUND:
                fails.append("%s: max|w| %.3e" % (name, r["w"]))
            if not r["solves"][0] > 0:
                fails.append("%s: hevi-ark2 did not run" % name)
            k = run(native, tmp, "rk")
            print("%-18s rk path (reference): max|u - a t|/(a t) %.3e, max|w| %.3e"
                  % ("", k["err"], k["w"]))
    print("FAILED: " + "; ".join(fails) if fails else "PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
