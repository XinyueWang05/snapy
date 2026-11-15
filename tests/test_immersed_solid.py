"""A column standing on a bottom-attached solid is the shorter wall column above it.

Column T = 1 - z/2, p = T^2, rho = T (z above the surface), g = 1, gamma 1.4, weno5 + lmars,
rk3, explicit, reflecting x1 walls, the default wb-wall-clamp. The wall column holds N cells
on the faces above the surface; the solid column holds the same faces plus k solid cells
below them (uniform: the same width; stretched: x1rat 1.05 at N = 32 and 1.05^(32/N) at N,
one map refined, the solid widths continuing the grid downwards). The fluid starts discretely
balanced (snapy.balance_column). Rest: 100 steps; pulse: an isentropic 1e-3 bump at z = .4,
to t = .3. Solid fill: the earth defaults (rho 1e3, p 1e9) or the surface cell's own values.
Gate: the fluid of the solid column equals the wall column (uniform and stretched rest: 1e-12;
stretched pulse: falls at order >= 1.8 from 128 to 256 cells, set by the grid-continuing solid
widths, mirrored widths match to round-off), and the dx-weighted fluid mass equals the wall
column's to 1e-13.
usage: python test_immersed_solid.py [--quick]"""
import math, os, sys, tempfile
import torch, yaml
import snapy
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

torch.set_default_dtype(torch.float64)
NG, GAM = 3, 1.4


def faces(n, k, stretched):
    """surface at 0, n fluid cells up to 1, k solid cells below (widths continue the grid)"""
    if not stretched:
        return torch.arange(-k, n + 1) / n
    r = 1.05 ** (32 / n)  # x1rat 1.05 at N = 32, the same map at every N
    h = (r - 1) / (r ** n - 1) * r ** torch.arange(-k, n).double()
    return torch.cat([torch.zeros(1), torch.cumsum(h, 0)]) - h[:k].sum()


def block(xf, internal=None):
    weno = {"type": "weno5", "scale": False, "shock": False}
    eos = {"type": "ideal-gas", "gammad": GAM, "density-floor": 1.e-10, "pressure-floor": 1.e-10,
           "limiter": False}
    c = {"geometry": {"type": "cartesian",
                      "bounds": {"x1min": float(xf[0]), "x1max": float(xf[-1]), "x2min": 0., "x2max": 1.,
                                 "x3min": 0., "x3max": 1.},
                      "cells": {"nx1": len(xf) - 1, "nx2": 1, "nx3": 1, "nghost": NG},
                      "x1f": [float(v) for v in xf]},
         "distribute": {"layout": "slab", "nb2": 1, "nb3": 1, "verbose": False},
         "dynamics": {"equation-of-state": eos, "reconstruct": {"vertical": weno, "horizontal": weno},
                      "riemann-solver": {"type": "lmars"}},
         "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                             "x2-inner": "periodic", "x2-outer": "periodic",
                                             "x3-inner": "periodic", "x3-outer": "periodic"}},
         "integration": {"type": "rk3", "cfl": 0.9, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9,
                         "ncycle_out": 0},
         "forcing": {"const-gravity": {"grav1": -1.}}}
    if internal:
        c["boundary-condition"]["internal"] = internal
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        yaml.safe_dump(c, open(p, "w"))
        b = MeshBlock(MeshBlockOptions.from_yaml(p))
    b.to(torch.device("cpu"), torch.float64)
    return b


def fluid(f, pulse):
    """balanced cell means of the column on faces f (z above the surface), plus the pulse"""
    h = f[1:] - f[:-1]
    t = lambda z: 1. - .5 * z
    col = torch.zeros(kIPR + 1, 1, 1, len(h))
    col[kIDN, 0, 0] = (t(f[:-1]) ** 2 - t(f[1:]) ** 2) / h
    col[kIPR, 0, 0] = (t(f[:-1]) ** 3 - t(f[1:]) ** 3) / (1.5 * h)
    stretched = bool((h.max() - h.min()) > 1e-12 * h.mean())
    try:
        wb = snapy.balance_column(col, h, 1., rtol=1.e-13, max_iter=400, stretched=stretched)[0]
    except RuntimeError:  # a fine stretched column stalls at round-off
        wb = snapy.balance_column(col, h, 1., rtol=1.e-12, max_iter=400, stretched=stretched)[0]
    if pulse:
        z = .5 * (f[1:] + f[:-1])
        d = 1.e-3 * torch.exp(-((z - .4) / .1) ** 2)
        wb[kIDN, 0, 0] *= 1 + d
        wb[kIPR, 0, 0] *= (1 + d) ** GAM
    return wb[[kIDN, kIPR], 0, 0]


def run(n, k, stretched, pulse, fill):
    f = faces(n, k, stretched)
    rp = fluid(f[k:], pulse)
    out = {}
    sd, sp = (1.e3, 1.e9) if fill == "earth" else (float(rp[0, 0]), float(rp[1, 0]))
    for name, xf, kk in (("wall", f[k:], 0), ("solid", f, k)):
        b = block(xf, {"solid-density": sd, "solid-pressure": sp} if kk else None)
        w = b.buffer("hydro.D").clone().zero_()
        for c, v in ((kIDN, rp[0]), (kIPR, rp[1])):
            w[c, 0, 0, NG + kk:NG + kk + n] = v
            w[c, 0, 0, :NG + kk] = v[0]
            w[c, 0, 0, NG + kk + n:] = v[-1]
        init = {"hydro_w": w}
        if kk:
            solid = torch.zeros_like(w[kIDN], dtype=torch.bool)
            solid[..., NG:NG + kk] = True
            w[kIDN][solid], w[kIPR][solid] = sd, sp
            init["solid"] = solid
        var, _ = b.initialize(init)
        h = xf[1:] - xf[:-1]
        if name == "wall":  # one dt and step count, from the wall column
            dt = .4 * float(h.min()) / math.sqrt(GAM * 1.)
            nstep = math.ceil(.3 / dt) if pulse else 100
            dt = .3 / nstep if pulse else dt
        for _ in range(nstep):
            for st in range(3):
                b.forward(var, dt, st)
        u = var["hydro_u"][:, 0, 0, NG + kk:NG + kk + n]
        wv = var["hydro_w"][:, 0, 0, NG + kk:NG + kk + n]
        out[name] = (u.clone(), wv.clone(), float((u[kIDN] * h[kk:]).sum()))
    (uw, ww, mw), (us, ws, ms) = out["wall"], out["solid"]
    return dict(diff=float((us - uw).abs().max()), wmax=float(ws[kIV1].abs().max()),
                wmax_wall=float(ww[kIV1].abs().max()), mass=abs(ms - mw) / mw,
                finite=bool(torch.isfinite(us).all()))


def main(quick=False):
    ok, rows = True, {}
    for stretched in (False, True):
        for pulse in (False, True):
            ns = (32, 64, 128) + ((256,) if stretched and pulse else ())
            for n in ns[:1] if quick else ns:
                for k in (1, 2, 6):
                    for fill in ("earth", "matched"):
                        r = run(n, k, stretched, pulse, fill)
                        rows[stretched, pulse, n, k, fill] = r
                        print(f"COL {'stretched' if stretched else 'uniform':9s} {'pulse' if pulse else 'rest':5s} "
                              f"N={n:3d} k={k} fill={fill:7s} max|u_solid-u_wall|={r['diff']:.2e} "
                              f"max|w| solid={r['wmax']:.2e} wall={r['wmax_wall']:.2e} "
                              f"dmass={r['mass']:.1e} finite={r['finite']}", flush=True)
                        exact = not stretched or not pulse
                        ok &= r["finite"] and r["mass"] <= 1.e-13 and (not exact or r["diff"] <= 1.e-12)
    if not quick:
        for k in (1, 2, 6):
            for fill in ("earth", "matched"):
                d = [rows[True, True, n, k, fill]["diff"] for n in (128, 256)]
                order = (math.log2(d[0] / d[1]) if d[1] > 0
                         else float("inf") if d[0] == 0 else math.nan)
                print(f"ORDER stretched pulse k={k} fill={fill}: 128->256 {order:.2f}", flush=True)
                ok &= math.isfinite(d[0]) and order >= 1.8
    print("GATE immersed solid column:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main("--quick" in sys.argv))
