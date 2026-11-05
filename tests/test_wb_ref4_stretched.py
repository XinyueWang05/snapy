#!/usr/bin/env python3
"""#280 item (4): SNAP_WB_REF4 (and SNAP_X1_CENTROID_EXACT, which implies it) on a keyed stretched x1.

On a keyed stretched x1 the kernel's cell reference pressure is already the cell average of the degree-5
interpolant through six faces (hydro_ref_x1_impl.h, #280 (iv)); REF4 keeps it there and contributes only its
density reference (dref = pref F(rho/p)) and face density. The switch must therefore not move the rest state.

  gate   an isothermal column on an x1rat 1.058 deck (nx1 32 and 64, 300 km, g 22, H 107 km) is balanced by
         snapy.balance_column(..., stretched=True) in three processes: switch unset, SNAP_WB_REF4=1 and
         SNAP_X1_CENTROID_EXACT=1. The balanced columns must be BITWISE equal across the three. The balanced
         column then runs one RK3 step in the solver on the same deck; its rest residual
         max |rho v1| / (dt g <rho>) must stay below REST_TOL in every arm.
  order  the neutral-polytrope oracle of test_wb_ref4_order.py (one RK3 step, N2_eff of an overturning mode),
         on a keyed stretched family at a fixed h_max / h_min = RATIO (x1rat = RATIO^(1/(nz-1))), nz 32/64/128,
         1 and 3 e-folds: with the switch on the observed order of |N2_eff| is >= ORDER_ON over both doublings.
         The index-unit filter F makes the stretched density reference third order (fourth on a uniform grid).
The switch is read once per process, so each arm runs in its own process.

  python test_wb_ref4_stretched.py [--device cpu|cuda]
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ARMS = (None, "SNAP_WB_REF4", "SNAP_X1_CENTROID_EXACT")
GATE_NZ = (32, 64)
GATE_Q = 1.058
GATE_L, GATE_G, GATE_H = 3.0e5, 22.0, 1.07e5
REST_TOL = 1.0e-12
NZS = (32, 64, 128)
EFOLDS = (1.0, 3.0)
RATIO = 4.0
ORDER_ON = 2.5
AMP = 1.0e-4
GAMMA = 5.0 / 3.0
M = 1.0 / (GAMMA - 1.0)
G = M + 1.0
NG, IV1, IV2 = 3, 1, 2
GLX, GLW = np.polynomial.legendre.leggauss(4)


def eos():
    return {"type": "ideal-gas", "gammad": GAMMA, "weight": 8.31446, "density-floor": 1.e-12,
            "pressure-floor": 1.e-12, "temperature-floor": 1.e-12, "limiter": True}


def config(nz, lz, q, nx2, grav):
    return {"geometry": {"type": "cartesian",
                         "bounds": {"x1min": 0.0, "x1max": lz, "x2min": 0.0, "x2max": 2.0 * lz,
                                    "x3min": 0.0, "x3max": 1.0},
                         "cells": {"nx1": nz, "nx2": nx2, "nx3": 1, "nghost": NG, "x1rat": q}},
            "dynamics": {"equation-of-state": eos(),
                         "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                                         "horizontal": {"type": "weno5", "scale": True, "shock": False}},
                         "riemann-solver": {"type": "lmars"}},
            "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                                "x2-inner": "periodic", "x2-outer": "periodic",
                                                "x3-inner": "periodic", "x3-outer": "periodic"}},
            "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
            "forcing": {"const-gravity": {"grav1": -grav, "gravity-work": "face"}}}


def make_block(cfg, device):
    import yaml
    from snapy import MeshBlock, MeshBlockOptions
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device), torch.float64)
    return block


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def gate(nz, device):
    """balanced column (hash) and the solver's one-step rest residual"""
    import snapy
    from snapy import kIDN, kIPR
    block = make_block(config(nz, GATE_L, GATE_Q, 1, GATE_G), device)
    dx1f = block.buffer("coord.dx1f")[NG:-NG].cpu().contiguous()
    zc = block.buffer("coord.x1v")[NG:-NG].cpu()
    col = torch.zeros((5, 1, 1, nz), dtype=torch.float64)
    col[kIDN, 0, 0] = 1.e7 / (GATE_G * GATE_H) * torch.exp(-zc / GATE_H)
    col[kIPR, 0, 0] = 1.e7 * torch.exp(-zc / GATE_H)
    wb, err, sweeps = snapy.balance_column(col, dx1f, GATE_G, True, 1.e-13, 400, "cartesian", True)
    w = torch.zeros((5, 1, 1, nz + 2 * NG), dtype=torch.float64)
    w[..., NG:-NG] = wb
    w[..., :NG] = wb[..., :1]
    w[..., -NG:] = wb[..., -1:]
    bv, _ = block.initialize({"hydro_w": w.to(device)})
    rho = bv["hydro_u"][kIDN].clone()
    dt = 0.3 * float(dx1f.min()) / math.sqrt(GAMMA * float((wb[kIPR] / wb[kIDN]).max()))
    block.inc_cycle()
    for st in range(len(block.intg.stages)):
        block.forward(bv, dt, st)
    assert block.check_redo(bv) == 0
    u = bv["hydro_u"]
    rest = float((u[IV1] / (dt * GATE_G * rho)).abs()[..., NG:-NG].max())
    return {"column": sha(wb), "residual": err, "sweeps": sweeps, "rest": rest, "state": sha(u)}


def cell_average_prims(zf, xf, lz, amp):
    """exact (4x4 Gauss-Legendre) cell averages of rho, m1, m2, E -> primitives (test_wb_ref4_order.py)"""
    q, k = math.pi / lz, math.pi / lz
    zc, hz = 0.5 * (zf[1:] + zf[:-1]), 0.5 * (zf[1:] - zf[:-1])
    xc, hx = 0.5 * (xf[1:] + xf[:-1]), 0.5 * (xf[1:] - xf[:-1])
    acc = np.zeros((4, len(xc), len(zc)))
    for a, wa in zip(GLX, GLW):
        for b, wb in zip(GLX, GLW):
            z = (zc + a * hz)[None, :]
            x = (xc + b * hx)[:, None]
            t0 = 1.0 + lz - z
            rho, p = t0 ** M, t0 ** (M + 1.0)
            m1 = amp * k * np.sin(q * z) * np.cos(k * x)
            m2 = -amp * q * np.cos(q * z) * np.sin(k * x)
            e = p / (GAMMA - 1.0) + (m1 * m1 + m2 * m2) / (2.0 * rho)
            acc += 0.25 * wa * wb * np.stack(np.broadcast_arrays(rho, m1, m2, e))
    rho, m1, m2, e = acc
    return rho, m1 / rho, m2 / rho, (GAMMA - 1.0) * (e - (m1 * m1 + m2 * m2) / (2.0 * rho))


def n2_eff(nz, efolds, device):
    from snapy import kIDN, kIPR
    lz = math.exp(efolds / (M + 1.0)) - 1.0
    q = RATIO ** (1.0 / (nz - 1))
    inner = slice(NG, -NG)
    ds = []
    for amp in (0.0, AMP):
        block = make_block(config(nz, lz, q, 2 * nz, G), device)
        zf = block.buffer("coord.x1f").cpu().numpy()
        xf = block.buffer("coord.x2f").cpu().numpy()
        dt = 0.3 * float(np.diff(zf[NG:-NG]).min()) / math.sqrt(GAMMA * (1.0 + lz))
        w0 = torch.zeros((5, 1, len(xf) - 1, len(zf) - 1), dtype=torch.float64)
        rho, v1, v2, p = cell_average_prims(zf, xf, lz, amp)
        for c, v in ((kIDN, rho), (IV1, v1), (IV2, v2), (kIPR, p)):
            w0[c, 0] = torch.from_numpy(v)
        bv, _ = block.initialize({"hydro_w": w0.to(device)})
        ev = block.module("hydro.eos")
        s = []
        for _ in range(2):
            w = ev.compute("U->W", [bv["hydro_u"]]).cpu()[:, 0, inner, inner]
            s.append(torch.log(w[kIPR]) - GAMMA * torch.log(w[kIDN]))
            if len(s) == 1:
                block.inc_cycle()
                for st in range(len(block.intg.stages)):
                    block.forward(bv, dt, st)
                assert block.check_redo(bv) == 0
        ds.append(s[1] - s[0])
    vz = w0[IV1, 0, inner, inner]
    return float(G / GAMMA * (-(ds[1] - ds[0]) / dt * vz).sum() / (vz * vz).sum())


def child(device, order):
    out = {"gate": {str(nz): gate(nz, device) for nz in GATE_NZ}}
    if order:
        out["order"] = {str(n): [n2_eff(nz, n, device) for nz in NZS] for n in EFOLDS}
    return out


def run(arm, device, tmpdir):
    env = dict(os.environ)
    for a in ARMS[1:]:
        env.pop(a, None)
    env["SNAP_FLUX_COVARIANCE"] = "1"
    if arm is not None:
        env[arm] = "1"
    out = os.path.join(tmpdir, f"arm_{arm}.json")
    cmd = [sys.executable, __file__, "--child", "--out", out, "--device", device]
    if arm == "SNAP_WB_REF4":
        cmd.append("--order")
    r = subprocess.run(cmd, env=env)
    return json.load(open(out)) if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--order", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available")
        return 125
    if a.child:
        json.dump(child(a.device, a.order), open(a.out, "w"))
        return 0
    failures = []
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        res = {arm: run(arm, a.device, tmp) for arm in ARMS}
    for arm, r in res.items():
        if r is None:
            failures.append(f"{arm or 'switch unset'}: the stretched deck or balance_column failed")
    if not failures:
        off = res[None]["gate"]
        for arm in ARMS:
            for nz, g in res[arm]["gate"].items():
                same = g["column"] == off[nz]["column"]
                print(f"gate nx1 {nz}, {arm or 'switch unset'}: residual {g['residual']:.2e} "
                      f"({g['sweeps']} sweeps), column {'bitwise equal to' if same else 'DIFFERS from'} "
                      f"the switch-off column, rest residual {g['rest']:.2e}, end state "
                      f"{'bitwise equal' if g['state'] == off[nz]['state'] else 'differs'}", flush=True)
                if not same:
                    failures.append(f"gate nx1 {nz}, {arm}: the switch moved the balanced column")
                if not g["rest"] < REST_TOL:
                    failures.append(f"gate nx1 {nz}, {arm or 'unset'}: rest residual {g['rest']:.2e}")
        for n, v in res["SNAP_WB_REF4"]["order"].items():
            order = [math.log2(abs(v[i]) / abs(v[i + 1])) for i in range(len(v) - 1)]
            print(f"order, SNAP_WB_REF4, h_max/h_min {RATIO:g}, {float(n):g} e-folds: N2_eff "
                  + " / ".join(f"{x:+.3e}" for x in v) + " (nz " + "/".join(map(str, NZS))
                  + "), order " + ", ".join(f"{o:.2f}" for o in order), flush=True)
            if min(order) < ORDER_ON:
                failures.append(f"order, {float(n):g} e-folds: {min(order):.2f} < {ORDER_ON}")
    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("### wb_ref4 stretched test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
