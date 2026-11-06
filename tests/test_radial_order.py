#!/usr/bin/env python3
"""#280: x1 WENO5/cp5 on r^2-dr cell averages is fifth order on spherical-polar and gnomonic decks."""
import math, os, sys, tempfile
import numpy as np, torch, yaml, snapy

DEV = "cuda" if "--device" in sys.argv and "cuda" in sys.argv else "cpu"
if DEV == "cuda" and not torch.cuda.is_available(): sys.exit(125)
GX, GW = np.polynomial.legendre.leggauss(12)
f = lambda r: 1. + 0.5 * np.sin(4. * r) * np.exp(-0.5 * r)


def error(geom, kind, n, stretched):  # max face error of the deck's recon1 on exact r^2 averages
    gn, rc = geom == "gnomonic-equiangle", {"type": kind, "scale": False, "shock": False}
    cells = {"nx1": n, "nx2": 4 if gn else 1, "nx3": 4 if gn else 1, "nghost": 3, **({"x1rat": 1.058 ** (32. / n)} if stretched else {})}
    ang = {"x2min_pi": -.25, "x2max_pi": .25, "x3min_pi": -.25, "x3max_pi": .25} if gn else {"x2min": 1., "x2max": 2., "x3min": 0., "x3max": 1.}
    cfg = {"reference-state": {"Tref": 300., "Pref": 1.e5}, "species": [{"name": "dry", "composition": {"H": 1.5, "He": .15}, "cv_R": 2.5}],
           "geometry": {"type": geom, "bounds": {"x1min": 1., "x1max": 2., **ang}, "cells": cells},
           "dynamics": {"equation-of-state": {"type": "ideal-gas"}, "reconstruct": {"vertical": rc, "horizontal": rc}, "riemann-solver": {"type": "lmars"}},
           "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting", **({k: "custom" for k in ("x2-inner", "x2-outer", "x3-inner", "x3-outer")} if gn else {})}}}
    if gn: cfg["distribute"] = {"layout": "cubed-sphere", "nb2": 1, "nb3": 1, "blocks_per_process": 6}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fp:
        yaml.safe_dump(cfg, fp)
    opts = snapy.MeshOptions.from_yaml(fp.name)
    os.unlink(fp.name)
    if gn: opts.blocks_per_process(6)
    b = snapy.Mesh(opts).blocks[0]
    b.to(torch.device(DEV), torch.float64)
    xf = b.buffer("coord.x1f").cpu().double().numpy()
    r = 0.5 * (xf[:-1] + xf[1:])[:, None] + 0.5 * np.diff(xf)[:, None] * GX[None, :]
    wr = GW[None, :] * r * r
    w = torch.tensor((wr * f(r)).sum(1) / wr.sum(1)).view(1, 1, 1, -1).repeat(5, 1, 1, 1).to(DEV)
    out, i = b.module("hydro.recon1").forward(w, 3, False).cpu(), np.arange(6, n + 1)  # three faces off each wall
    return max(np.abs(out[s][0, 0, 0].numpy()[i] - f(xf[i])).max() for s in (0, 1))


fails = []
for geom, grids in (("spherical-polar", (False,)), ("gnomonic-equiangle", (False, True))):
    for kind in ("weno5", "cp5"):
        for st in grids:
            e = [error(geom, kind, n, st) for n in (32, 64, 128)]
            o = [math.log2(e[0] / e[1]), math.log2(e[1] / e[2])]
            case = f"{geom} {kind} {'stretched' if st else 'uniform'}"
            print(f"{case}: err {e[0]:.2e} {e[1]:.2e} {e[2]:.2e} orders {o[0]:.2f} {o[1]:.2f}")
            fails += [f"{case} order {o[1]:.2f} < 4.5"] * (o[1] < 4.5)
print("FAIL:", fails) if fails else print("### radial order test passed. ###")
sys.exit(1 if fails else 0)
