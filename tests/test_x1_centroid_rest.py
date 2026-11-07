#!/usr/bin/env python3
"""SNAP_X1_CENTROID_EXACT: a spherical-polar column of r^2 cell averages stays at rest.

The oracle of docs/derivations/x1-centroid-spherical.md, section 5. A hydrostatic column with g = 1 and a
linear density, rho = 1 - A (r - r0), p = 1 + int_r^{r0+2} rho dr (quadratic), on a spherical-polar shell
sector x1 = [r0, r0 + 2], reflecting walls, initialised with the EXACT cell averages a finite-volume cell holds:
the r^2-weighted means of rho and p (six-point Gauss-Legendre per cell, exact here). Every x1 formula of the
switch is exact for such a column, so its residual is round-off; a smooth non-polynomial column (an isothermal
exp(-(r - r0))) instead keeps the fourth-order truncation of the reference, independent of r0, as in Cartesian.
One RK3 step at dt = 0.3 dx1 / sqrt(1.4); the radial force imbalance is
    f = max |rho v1| / (dt g <rho>)
over the interior cells and over the three cells next to each wall. Without the switch the uniform-measure
x1 formulas leave an O(dx1^2 / r0) imbalance; with it the column is balanced to round-off. Asserted at
r0 = 5 and 1000, nz = 32, explicit (implicit-scheme 0) and vertically implicit (1), with the full pressure
force (non-hydrostatic 1) and in hydrostatic-split mode (non-hydrostatic 0, where the hydrostatic correction
must use the same r^2 pressure operator):
  switch on:  f < TOL_ON in the interior and at the walls, at every r0;
  switch off: f > TOL_OFF at r0 = 5, so an ignored or misspelled switch fails.
The switch is read once per process, so each arm runs in its own process.

  python test_x1_centroid_rest.py [--device cpu|cuda]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

NZ = 32
R0S = (5.0, 1000.0)
SCHEMES = (0, 1)
NHS = (1.0, 0.0)
LZ = 2.0
A = 0.3
TOL_ON = 1.0e-10
TOL_OFF = 1.0e-8
NG, IV1 = 3, 1
GLX, GLW = np.polynomial.legendre.leggauss(6)


def config(r0, scheme, nh, rat=1.0, nz=NZ):
    half = 0.05
    return {"geometry": {"type": "spherical-polar",
                         "bounds": {"x1min": r0, "x1max": r0 + LZ,
                                    "x2min": 0.5 * math.pi - half, "x2max": 0.5 * math.pi + half,
                                    "x3min": 0.0, "x3max": 2.0 * half},
                         "cells": {"nx1": nz, "nx2": 4, "nx3": 4, "nghost": NG,
                                   **({"x1rat": rat} if rat != 1.0 else {})}},
            "dynamics": {"equation-of-state": {"type": "ideal-gas", "gammad": 1.4, "weight": 8.31446,
                                               "density-floor": 1.e-12, "pressure-floor": 1.e-12,
                                               "temperature-floor": 1.e-12, "limiter": True},
                         "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                                         "horizontal": {"type": "weno5", "scale": True, "shock": False}},
                         "riemann-solver": {"type": "lmars"}},
            "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                                "x2-inner": "reflecting", "x2-outer": "reflecting",
                                                "x3-inner": "periodic", "x3-outer": "periodic"}},
            "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": scheme, "nlim": -1,
                            "tlim": 1.e9},
            "forcing": {"const-gravity": {"grav1": -1.0, "non-hydrostatic": nh}}}


def column(r, r0):
    """rho and p of the hydrostatic column (g = 1)"""
    x = r - r0
    return 1.0 - A * x, 1.0 + (LZ - x) - 0.5 * A * (LZ * LZ - x * x)


def r2_means(rf, r0):
    """the r^2-weighted cell averages of rho and p on cells [rf_i, rf_i+1]"""
    c, h = 0.5 * (rf[1:] + rf[:-1]), 0.5 * (rf[1:] - rf[:-1])
    r = c[:, None] + h[:, None] * GLX[None, :]
    w = GLW[None, :] * r * r
    return tuple((q * w).sum(1) / w.sum(1) for q in column(r, r0))


def make(cfg, device):
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


def face_order(kind, device):  # x1 recon on plain means of a stretched x1, N 32/64/128
    f, e = lambda r: 1.0 + 0.5 * np.sin(2.0 * r), []
    for nz in (32, 64, 128):
        cfg = config(R0S[0], 0, 1.0, 1.058 ** (32.0 / nz), nz)
        cfg["dynamics"]["reconstruct"]["vertical"]["type"] = kind
        block = make(cfg, device)
        rf = block.buffer("coord.x1f").cpu().numpy()
        r = 0.5 * (rf[1:] + rf[:-1])[:, None] + 0.5 * np.diff(rf)[:, None] * GLX[None, :]
        w = torch.tensor((GLW[None, :] * f(r)).sum(1) / 2.0).view(1, 1, 1, -1).repeat(5, 1, 1, 1)
        out, i = block.module("hydro.recon1").forward(w.to(device), 3, False).cpu(), np.arange(6, nz + 1)
        e.append(max(np.abs(out[s][0, 0, 0].numpy()[i] - f(rf[i])).max() for s in (0, 1)))
    return [math.log2(e[0] / e[1]), math.log2(e[1] / e[2])]


def imbalance(r0, scheme, nh, device, rat=1.0):
    from snapy import kIDN, kIPR
    block = make(config(r0, scheme, nh, rat), device)
    rf = block.buffer("coord.x1f").cpu().numpy()
    w = block.buffer("hydro.D").clone().zero_()
    # interior means; ghosts get the same formula (the reflecting walls refill them)
    rho, p = (torch.from_numpy(q).to(w) for q in r2_means(rf, r0))
    w[kIDN] = rho
    w[kIPR] = p
    bv, _ = block.initialize({"hydro_w": w})
    dt = 0.3 * (LZ / NZ if rat == 1.0 else float(np.diff(rf).min())) / math.sqrt(1.4)
    block.inc_cycle()
    for st in range(len(block.intg.stages)):
        block.forward(bv, dt, st)
    assert block.check_redo(bv) == 0
    u = bv["hydro_u"].cpu()[:, NG:-NG, NG:-NG, NG:-NG]
    f = (u[IV1] / (dt * u[kIDN])).abs().amax(dim=(0, 1))
    return {"interior": float(f[NG:-NG].max()), "wall": float(torch.cat((f[:NG], f[-NG:])).max())}


def child(device):
    out = {f"{r0:g}/{s}/{nh:g}/{rat:g}": imbalance(r0, s, nh, device, rat)
           for r0 in R0S for s in SCHEMES for nh in NHS for rat in (1.0, 1.058)}
    if os.environ.get("SNAP_X1_CENTROID_EXACT"):
        out["order"] = {k: face_order(k, device) for k in ("weno5", "cp5")}
    return out


def run(switch, device, tmpdir):
    env = dict(os.environ)
    env.pop("SNAP_X1_CENTROID_EXACT", None)
    if switch is not None:
        env["SNAP_X1_CENTROID_EXACT"] = switch
    out = os.path.join(tmpdir, f"rest_{switch}.json")
    subprocess.run([sys.executable, __file__, "--child", "--out", out, "--device", device],
                   env=env, check=True)
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available")
        return 125
    if a.child:
        json.dump(child(a.device), open(a.out, "w"))
        return 0
    failures = []
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        res = {s: run(s, a.device, tmp) for s in (None, "1")}
    for k, o in res["1"].pop("order").items():
        print(f"switch on, stretched x1 {k}: face orders {o[0]:.2f} {o[1]:.2f}")
        failures += [f"{k} face order {o[1]:.2f} < 4.5"] * (o[1] < 4.5)
    for key in res[None]:
        off, on = res[None][key], res["1"][key]
        r0, scheme, nh, rat = key.split("/")
        print(f"r0/H {r0:>5s} x1rat {rat} implicit-scheme {scheme} non-hydrostatic {nh}: "
              "max |rho v1|/(dt g rho), interior / wall: "
              f"off {off['interior']:.3e} / {off['wall']:.3e}  on {on['interior']:.3e} / {on['wall']:.3e}",
              flush=True)
        for part in ("interior", "wall"):
            if not on[part] < TOL_ON:
                failures.append(f"{key} {part}, switch on: {on[part]:.3e} >= {TOL_ON}")
        if float(r0) == R0S[0] and not off["interior"] > TOL_OFF:
            failures.append(f"{key}, switch off: {off['interior']:.3e} <= {TOL_OFF}, "
                            "the oracle does not tell the forms apart")
    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("### x1 centroid rest test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
