#!/usr/bin/env python3
"""#280 (4): REF4 on a keyed stretched x1 keeps the rest state bitwise.
The switch is read once per process, so each arm runs in its own. python test_wb_ref4_stretched.py [--device cuda]"""
import hashlib, json, math, os, subprocess, sys, tempfile
import torch, yaml

DEV = "cuda" if "--device" in sys.argv and "cuda" in sys.argv else "cpu"
GAM, ARMS = 5.0 / 3.0, (None, "SNAP_WB_REF4", "SNAP_X1_CENTROID_EXACT")


def block(nz, lz, q, nx2, g):
    from snapy import MeshBlock, MeshBlockOptions
    rc = {"type": "weno5", "scale": True, "shock": False}
    cfg = {"geometry": {"type": "cartesian", "bounds": {"x1min": 0., "x1max": lz, "x2min": 0., "x2max": 2 * lz, "x3min": 0.,
                        "x3max": 1.}, "cells": {"nx1": nz, "nx2": nx2, "nx3": 1, "nghost": 3, "x1rat": q}},
           "dynamics": {"equation-of-state": {"type": "ideal-gas", "gammad": GAM, "weight": 8.31446, "limiter": True,
                                               "density-floor": 1.e-12, "pressure-floor": 1.e-12, "temperature-floor": 1.e-12},
                        "reconstruct": {"vertical": rc, "horizontal": rc}, "riemann-solver": {"type": "lmars"}},
           "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                               "x2-inner": "periodic", "x2-outer": "periodic",
                                               "x3-inner": "periodic", "x3-outer": "periodic"}},
           "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0},
           "forcing": {"const-gravity": {"grav1": -g, "gravity-work": "face"}}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    b = MeshBlock(MeshBlockOptions.from_yaml(f.name)); os.unlink(f.name); b.to(torch.device(DEV), torch.float64)
    return b


def step(b, w, dt):  # one RK3 step from primitives w; returns (u0, u1)
    v, _ = b.initialize({"hydro_w": w.to(DEV)}); u0 = v["hydro_u"].clone(); b.inc_cycle()
    for st in range(len(b.intg.stages)):
        b.forward(v, dt, st)
    return u0, v["hydro_u"]


def gate(nz):  # balanced isothermal column (hash) and the rest residual of one RK3 step
    import snapy
    b = block(nz, 3.e5, 1.058, 1, 22.); dz = b.buffer("coord.dx1f")[3:-3].cpu(); z = b.buffer("coord.x1v")[3:-3].cpu()
    col = torch.zeros((5, 1, 1, nz), dtype=torch.float64)
    col[0, 0, 0], col[4, 0, 0] = 1.e7 / (22. * 1.07e5) * torch.exp(-z / 1.07e5), 1.e7 * torch.exp(-z / 1.07e5)
    wb, _, _ = snapy.balance_column(col, dz.contiguous(), 22., True, 1.e-13, 400, "cartesian", True)
    w = torch.cat([wb[..., :1]] * 3 + [wb] + [wb[..., -1:]] * 3, -1)
    dt = 0.3 * float(dz.min()) / math.sqrt(GAM * float((wb[4] / wb[0]).max()))
    u0, u1 = step(b, w, dt)
    rest = float((u1[1] / (dt * 22. * u0[0])).abs()[..., 3:-3].max())
    return hashlib.sha256(wb.numpy().tobytes()).hexdigest(), rest


if "--child" in sys.argv:
    print("RESULT " + json.dumps({"gate": [gate(nz) for nz in (32, 64)]})); sys.exit(0)
res = {}
for arm in ARMS:
    env = {k: v for k, v in os.environ.items() if k not in ARMS[1:]} | ({arm: "1"} if arm else {}) | {"SNAP_FLUX_COVARIANCE": "1"}
    p = subprocess.run([sys.executable, __file__, "--child"] + sys.argv[1:], env=env, capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    res[arm] = json.loads(lines[0][7:]) if lines else None
    print(arm, "refused or failed" if res[arm] is None else res[arm])
fails = [f"{a}: refused or failed" for a in ARMS if res[a] is None]
if not fails:
    fails += [f"{a}: switch moved the balanced column" for a in ARMS if [g[0] for g in res[a]["gate"]] != [g[0] for g in res[None]["gate"]]]
    fails += [f"{a}: rest residual {g[1]}" for a in ARMS for g in res[a]["gate"] if not g[1] < 1.e-12]
print("FAIL:", fails) if fails else print("### wb_ref4 stretched test passed. ###")
sys.exit(1 if fails else 0)
