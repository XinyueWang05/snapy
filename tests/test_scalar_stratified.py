#!/usr/bin/env python3
"""S177: passive-scalar order on a stratified column, uniform r and the limiter bound. python test_scalar_stratified.py [--device cuda]"""
import math, os, sys, tempfile
import numpy as np, torch, yaml
from snapy import MeshBlock, MeshBlockOptions, kIDN

DEV = "cuda" if "cuda" in sys.argv else "cpu"
A, RS, ZC = 22.0 / (3.5 * 3937.0 * 600.0), 1.0e7 / (3937.0 * 600.0), 1.5e5  # g/(cp Ts), rho_s, mid column
M = lambda z: RS / (3.5 * A) * (1 - (1 - A * z) ** 3.5)  # mass coordinate of the dry adiabat
Z = lambda m: (1 - (1 - 3.5 * A * m / RS) ** (1 / 3.5)) / A
GX, GW = np.polynomial.legendre.leggauss(16)


def run(n, recon, r0, ub=None, scale=True):  # snapy's Scalar.forward, mass flux F0 in hydro.F1; r carried exactly along m
    sc = {"nvar": 1, "reconstruct": {"type": recon, "scale": scale, "shock": True}}
    if ub: sc["upper-bound"] = ub
    rc = {"type": recon, "scale": True, "shock": True}
    cfg = {"species": [{"name": "dry", "composition": {"H": 1.5, "He": 0.15}, "cv_R": 2.5}], "scalar": sc,
           "geometry": {"type": "cartesian", "bounds": {"x1min": 0., "x1max": 3.e5, "x2min": 0., "x2max": 1., "x3min": 0.,
                        "x3max": 1.}, "cells": {"nx1": n, "nx2": 1, "nx3": 1, "nghost": 3}},
           "dynamics": {"equation-of-state": {"type": "ideal-gas", "limiter": True},
                        "reconstruct": {"vertical": rc, "horizontal": rc}, "riemann-solver": {"type": "hllc"}},
           "forcing": {"const-gravity": {"grav1": -22.}},
           "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting"}}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    b = MeshBlock(MeshBlockOptions.from_yaml(f.name)); os.unlink(f.name); b.to(torch.device(DEV), torch.float64)
    xf = b.buffer("coord.x1f").cpu().numpy(); mf = M(xf); h = np.diff(xf); f0 = RS * (1 - A * ZC) ** 2.5 * 0.5
    qbar = lambda t: (GW * r0(0.5 * (mf[:-1, None] + mf[1:, None]) + 0.5 * np.diff(mf)[:, None] * GX - f0 * t)
                      ).sum(1) / 2 * np.diff(mf) / h
    u = b.buffer("hydro.D").clone().zero_(); u[kIDN] = torch.tensor(np.diff(mf) / h).view(1, 1, -1); b.buffer("hydro.F1")[kIDN].fill_(f0)
    dt = 0.05 * (h * np.diff(mf) / h / f0)[3:-3].min(); ns = math.ceil(21600 / dt); dt = 21600 / ns
    L = lambda q, t: b.module("scalar").forward(dt, torch.cat([torch.tensor(qbar(t)[:3]).to(q), q[0, 0, 0, 3:-3], torch.tensor(qbar(t)[-3:]).to(q)]).view(q.shape), {"hydro_u": u})
    q, t = torch.tensor(qbar(0.0)).view(1, 1, 1, -1).to(u), 0.0
    for _ in range(ns):
        q1 = q + L(q, t); q2 = .75 * q + .25 * (q1 + L(q1, t + dt)); q = q / 3 + 2 / 3 * (q2 + L(q2, t + dt / 2)); t += dt
    return q.cpu().numpy()[0, 0, 0, 3:-3], qbar(t)[3:-3], (np.diff(mf) / h)[3:-3], h[3:-3]


fails, sine = [], lambda m: 1 + 0.5 * np.sin(2 * np.pi * Z(m) / 1.5e5)
for recon in ("cp5", "weno5"):  # RED/GREEN: L1 order against the exact answer >= 4.5 (q/rho of cell averages: 2)
    e = [(lambda q, x, rho, h: (abs(q - x) * h).sum() / (abs(x - rho) * h).sum())(*run(n, recon, sine)) for n in (50, 100, 200)]
    o = [math.log2(e[i] / e[i + 1]) for i in (0, 1)]; print(recon, "L1", e, "order", o)
    fails += [f"{recon} order {o}"] * int(min(o) < 4.5)
    for lo in (0.2, 0.0):  # gate: 0 <= r <= upper-bound; weno5 dips 5.56e-4 below 0.2, not gated
        q, _, rho, _ = run(64, recon, lambda m: np.where(Z(m) < ZC - 3.e4, lo, 1.0), ub=1.0); r = q / rho
        print(recon, "step", lo, "-> 1, upper-bound 1: min r", r.min(), "max r - 1", r.max() - 1)
        fails += [f"{recon} step {lo}: r in [{r.min()}, {r.max()}]"] * int(r.min() < -1e-12 or r.max() > 1 + 1e-12)
for recon, scale in [(a, b) for a in ("cp5", "weno5") for b in (False, True)]:  # a uniform r stays uniform
    q, _, rho, _ = run(64, recon, lambda m: 0.3 + 0 * m, scale=scale); d = abs(q / rho / 0.3 - 1).max()
    print(recon, "uniform r, scale", scale, "max |r/r0 - 1|", d); fails += [f"{recon} {scale} uniform r drifts {d}"] * int(d > 1e-12)
print("FAIL:", fails) if fails else print("### scalar stratified test passed. ###")
sys.exit(1 if fails else 0)
