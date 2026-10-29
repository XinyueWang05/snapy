#!/usr/bin/env python3
"""Conservation and forcing through the hevi-ark2 (ARS(2,3,2)) step, the default of
implicit-scheme 9 under rk3 (no implicit-integrator key).

scalars  A dry isothermal column (x1 vertical, 32 levels of 1 km, reflecting walls; x2
         periodic, 8 cells of 2 km) in the solver's discrete balance, with a vertical wave
         w = 5 sin(pi z / H), a sheared wind u = 10 + 5 cos(pi z / H) and a 0.5 K thermal
         blob, carries a uniform tracer R0 and a blob tracer. After NSTEP_S steps: total
         tracer mass sum(s V) of each tracer changes by <= TOL relative; the uniform tracer
         stays uniform, max |s / (rho R0) - 1| <= TOL; the blob tracer moved.
forcing  A dry isothermal column at rest (40 levels of 1 km; x2 periodic, 4 cells) pushed by
         a uniform horizontal acceleration a; exact answer u(t) = a t, w = 0. Two arms:
         native const-gravity grav2 = a (a native forcing, in the vertical part V), and a
         TorchScript user stage forcing hydro_du[IVY] = dt rho a with its work (explicit, in
         H). Each: max |u - a t| / (a t) <= TOL_U after NSTEP_F steps of fixed dt, max |w| <=
         W_BOUND. The rk path (implicit-integrator: rk) is reported for reference.

  python test_hevi_conservation.py
"""
import math
import os
import sys
import tempfile

import torch
import yaml

G, T0, PS, RD = 10.0, 300.0, 1.0e5, 287.0
NZ_S, DZ_S, NX2_S, DX_S = 32, 1.0e3, 8, 2.0e3
NSTEP_S, R0, TOL = 40, 1.0e-3, 1.0e-12
NZ_F, DZ_F, NX2_F, DX_F = 40, 1.0e3, 4, 10.0e3
A, NSTEP_F, DT = 1.0e-3, 50, 20.0
# W_BOUND: 3.5x the rk path's 2.9e-8 on the same box (the column's balance residual)
TOL_U, W_BOUND = 1.0e-10, 1.0e-7


def scalar_config():
    return {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry", "composition": {"N": 1.56, "O": 0.42, "Ar": 0.01},
                     "cv_R": 2.5}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": NZ_S * DZ_S, "x2min": 0., "x2max": NX2_S * DX_S,
                                "x3min": 0., "x3max": DX_S},
                     "cells": {"nx1": NZ_S, "nx2": NX2_S, "nx3": 1, "nghost": 3}},
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


def scalars(fails):
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

    torch.set_default_dtype(torch.float64)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "deck.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(scalar_config(), f)
        block = MeshBlock(MeshBlockOptions.from_yaml(p))
    block.to(torch.device("cpu"))

    ng = 3
    z = (torch.arange(NZ_S, dtype=torch.float64) + 0.5) * DZ_S
    x = (torch.arange(NX2_S, dtype=torch.float64) + 0.5) * DX_S
    pz = PS * torch.exp(-G * z / (RD * T0))
    col = torch.zeros(kIPR + 1, 1, 1, NZ_S, dtype=torch.float64)
    col[kIDN, 0, 0], col[kIPR, 0, 0] = pz / (RD * T0), pz
    wb, _, _ = snapy.balance_column(col, torch.full((NZ_S,), DZ_S, dtype=torch.float64), G,
                                    rtol=1.e-12, max_iter=400)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ_S] = wb[c, 0, 0]
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ_S:] = w[c][..., ng + NZ_S - 1:ng + NZ_S]
    H = NZ_S * DZ_S
    zz = torch.zeros(w.shape[-1])
    zz[ng:ng + NZ_S] = z
    xx = torch.zeros(w.shape[-2])
    xx[ng:ng + NX2_S] = x
    blob = torch.exp(-((xx.view(-1, 1) - 0.3 * NX2_S * DX_S) / (2 * DX_S)) ** 2
                     - ((zz.view(1, -1) - 0.4 * H) / (0.15 * H)) ** 2)
    w[kIDN] = w[kIDN] / (1.0 + 0.5 / T0 * blob)  # +0.5 K at fixed p
    w[kIV1] = 5.0 * torch.sin(math.pi * zz / H).view(1, 1, -1)
    w[kIV1 + 1] = (10.0 + 5.0 * torch.cos(math.pi * zz / H)).view(1, 1, -1)
    r = torch.full((2,) + tuple(w.shape[1:]), R0, dtype=torch.float64)
    r[1] = R0 * (1.0 + blob)
    v, _ = block.initialize({"hydro_w": w, "scalar_r": r})

    I = (slice(None), 0, slice(ng, ng + NX2_S), slice(ng, ng + NZ_S))
    vol = DX_S * DX_S * DZ_S  # uniform Cartesian cells
    mass0 = (v["scalar_s"][I] * vol).sum(dim=(-1, -2))
    blob0 = v["scalar_s"][1][I[1:]].clone()
    for _ in range(NSTEP_S):
        dt = block.max_time_step(v)
        for s in range(len(block.module("intg").stages)):
            block.forward(v, dt, s)
        assert block.check_redo(v) == 0
    mass = (v["scalar_s"][I] * vol).sum(dim=(-1, -2))
    rel = ((mass - mass0) / mass0).abs()
    unif = (v["scalar_s"][0][I[1:]] / (v["hydro_u"][kIDN][I[1:]] * R0) - 1.0).abs().max().item()
    moved = (v["scalar_s"][1][I[1:]] - blob0).abs().max().item() / blob0.abs().max().item()
    name = block.options.hydro().icorr().integrator()
    print("scalars: integrator %s" % name)
    print("tracer mass relative change: uniform %.3e, blob %.3e (tol %.0e)"
          % (rel[0].item(), rel[1].item(), TOL))
    print("uniform tracer max |s/(rho R0) - 1| = %.3e; blob moved by %.3e of its max"
          % (unif, moved))
    if name != "hevi-ark2":
        fails.append("scalars: integrator %s, expected hevi-ark2" % name)
    if not rel.max().item() <= TOL:
        fails.append("tracer mass not conserved: %.3e" % rel.max().item())
    if not unif <= TOL:
        fails.append("uniform tracer departed: %.3e" % unif)
    if not moved > 1.e-3:
        fails.append("blob did not move (%.3e)" % moved)


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


def forcing_config(native, integrator=None):
    grav = {"grav1": -G}
    if native:
        grav["grav2"] = A
    return {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry", "composition": {"N": 1.56, "O": 0.42, "Ar": 0.01},
                     "cv_R": 2.5}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": NZ_F * DZ_F, "x2min": 0., "x2max": NX2_F * DX_F,
                                "x3min": 0., "x3max": DX_F},
                     "cells": {"nx1": NZ_F, "nx2": NX2_F, "nx3": 1, "nghost": 3}},
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
        yaml.safe_dump(forcing_config(native, integrator), f)
    block = MeshBlock(MeshBlockOptions.from_yaml(p))
    block.to(torch.device("cpu"))
    if not native:
        path = os.path.join(tmp, "push.pt")
        torch.jit.script(ConstantPush(A).eval()).save(path)
        block.set_user_stage_forcings([path])

    ng = 3
    z = (torch.arange(NZ_F, dtype=torch.float64) + 0.5) * DZ_F
    pz = PS * torch.exp(-G * z / (RD * T0))
    col = torch.zeros(kIPR + 1, 1, 1, NZ_F, dtype=torch.float64)
    col[kIDN, 0, 0], col[kIPR, 0, 0] = pz / (RD * T0), pz
    wb, _, _ = snapy.balance_column(col, torch.full((NZ_F,), DZ_F, dtype=torch.float64), G,
                                    rtol=1.e-12, max_iter=400)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ_F] = wb[c, 0, 0]
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ_F:] = w[c][..., ng + NZ_F - 1:ng + NZ_F]
    v, _ = block.initialize({"hydro_w": w})
    for _ in range(NSTEP_F):
        for s in range(len(block.module("intg").stages)):
            block.forward(v, DT, s)
    I = (0, slice(ng, ng + NX2_F), slice(ng, ng + NZ_F))
    # velocities from the conserved state: hydro_w is refreshed at the start of each
    # stage's forward, so after a step it holds the last stage's input (t + dt/2 for
    # rk3's third stage, t + dt for ARS(2,3,2)), not u^{n+1}
    hu = v["hydro_u"]
    u = (hu[kIV1 + 1] / hu[kIDN])[I]
    wv = (hu[kIV1] / hu[kIDN])[I]
    exact = A * NSTEP_F * DT
    return dict(err=((u - exact).abs().max() / exact).item(), w=wv.abs().max().item(),
                name=block.options.hydro().icorr().integrator())


def forcing(fails):
    with tempfile.TemporaryDirectory() as tmp:
        for name, native in (("native grav2 (V)", True), ("user forcing (H)", False)):
            r = run(native, tmp)
            print("%-18s %s: max|u - a t|/(a t) %.3e (tol %.0e), max|w| %.3e"
                  % (name, r["name"], r["err"], TOL_U, r["w"]))
            if not r["err"] <= TOL_U:
                fails.append("%s: u error %.3e" % (name, r["err"]))
            if not r["w"] <= W_BOUND:
                fails.append("%s: max|w| %.3e" % (name, r["w"]))
            if r["name"] != "hevi-ark2":
                fails.append("%s: integrator %s, expected hevi-ark2" % (name, r["name"]))
            k = run(native, tmp, "rk")
            print("%-18s rk path (reference): max|u - a t|/(a t) %.3e, max|w| %.3e"
                  % ("", k["err"], k["w"]))


def main():
    torch.set_default_dtype(torch.float64)
    fails = []
    scalars(fails)
    forcing(fails)
    print("FAILED: " + "; ".join(fails) if fails else "PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
