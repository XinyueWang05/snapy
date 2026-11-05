#!/usr/bin/env python3
"""#280 (4): REF4 on a keyed stretched x1 keeps the rest state bitwise; its density reference is >= 2.5 order.
The switch is read once per process, so each arm runs in its own. python test_wb_ref4_stretched.py [--device cuda]"""
import hashlib, json, math, os, subprocess, sys, tempfile
import numpy as np, torch, yaml

DEV = "cuda" if "--device" in sys.argv and "cuda" in sys.argv else "cpu"
GAM, ARMS = 5.0 / 3.0, (None, "SNAP_WB_REF4", "SNAP_X1_CENTROID_EXACT")
GX, GW = np.polynomial.legendre.leggauss(4)


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


def n2(nz, ef):  # N2_eff of a seeded overturning mode in a neutral polytrope (test_wb_ref4_order.py), h_max/h_min 4
    m, g = 1 / (GAM - 1), 1 / (GAM - 1) + 1
    lz = math.exp(ef / g) - 1; k = math.pi / lz; ds = []
    for amp in (0.0, 1.e-4):
        b = block(nz, lz, 4.0 ** (1 / (nz - 1)), 2 * nz, g)
        zf, xf = b.buffer("coord.x1f").cpu().numpy(), b.buffer("coord.x2f").cpu().numpy()
        zc, hz, xc, hx = (zf[1:] + zf[:-1]) / 2, np.diff(zf) / 2, (xf[1:] + xf[:-1]) / 2, np.diff(xf) / 2
        acc = 0
        for a, wa in zip(GX, GW):
            for c, wc in zip(GX, GW):
                z, x = (zc + a * hz)[None], (xc + c * hx)[:, None]; t0 = 1 + lz - z
                m1, m2 = amp * k * np.sin(k * z) * np.cos(k * x), -amp * k * np.cos(k * z) * np.sin(k * x)
                acc = acc + 0.25 * wa * wc * np.stack(np.broadcast_arrays(t0 ** m, m1, m2, t0 ** (m + 1) * m + (m1 ** 2 + m2 ** 2) / (2 * t0 ** m)))
        w = torch.zeros((5, 1, len(xc), len(zc)), dtype=torch.float64)
        w[0, 0], w[1, 0], w[2, 0] = map(torch.from_numpy, (acc[0], acc[1] / acc[0], acc[2] / acc[0]))
        w[4, 0] = torch.from_numpy((GAM - 1) * (acc[3] - (acc[1] ** 2 + acc[2] ** 2) / (2 * acc[0])))
        dt = 0.3 * float(np.diff(zf[3:-3]).min()) / math.sqrt(GAM * (1 + lz))
        s = [torch.log(p[4]) - GAM * torch.log(p[0]) for p in (b.module("hydro.eos").compute("U->W", [u]).cpu()[:, 0, 3:-3, 3:-3] for u in step(b, w, dt))]
        ds.append(s[1] - s[0]); vz = w[1, 0, 3:-3, 3:-3]
    return float(g / GAM * (-(ds[1] - ds[0]) / dt * vz).sum() / (vz * vz).sum())


if "--child" in sys.argv:
    out = {"gate": [gate(nz) for nz in (32, 64)]}
    if os.environ.get("SNAP_WB_REF4"):
        out["n2"] = [[n2(nz, ef) for nz in (32, 64, 128)] for ef in (1.0, 3.0)]
    print("RESULT " + json.dumps(out)); sys.exit(0)
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
    o = [math.log2(abs(v[i] / v[i + 1])) for v in res["SNAP_WB_REF4"]["n2"] for i in (0, 1)]
    print("REF4 density-reference order (1 and 3 e-folds)", o)
    fails += [f"order {min(o)} < 2.5"] * int(min(o) < 2.5)
print("FAIL:", fails) if fails else print("### wb_ref4 stretched test passed. ###")
sys.exit(1 if fails else 0)
