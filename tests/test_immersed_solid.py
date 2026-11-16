"""Immersed solids against walls (every row fails on main 0f95515a).

COL: a column on a bottom-attached solid of k cells equals the shorter wall column above it
(T = 1 - z/2, p = T^2, rho = T, g = 1, weno5 + lmars, rk3, balanced start; rest 100 steps,
pulse to t = .3; uniform, and stretched with grid-continuing solid widths), fill earth
(1e3, 1e9) or matched: fluid to 1e-12 (stretched pulse: order >= 1.8, 128 -> 256 cells),
dx-weighted mass to 1e-13. THIN: a full-height x2 solid (no fluid cell) 1-2 cells wide between
fluid columns, gravity, explicit and scheme 9, equals the columns between reflecting walls.
TRACER: a uniform tracer over a step stays uniform, its solid fill notwithstanding.
X1 SPLIT: the pulsed solid column on two x1 blocks of one snapy.Mesh equals one block bit for
bit over 20 steps (uniform, stretched); implicit-scheme 9 with a solid refuses nb1 = 2 at initialize.
MPS: uniform COL arms in float32 to 1e-5. usage: python test_immersed_solid.py [--device mps]"""
import math, os, sys, tempfile, time
import torch, yaml
import snapy
from snapy import Mesh, MeshBlock, MeshBlockOptions, MeshOptions, kIDN, kIPR, kIV1

torch.set_default_dtype(torch.float64)
NG, GAM = 3, 1.4
DEV = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
DT, TOL = (torch.float32, 1.e-5) if DEV == "mps" else (torch.float64, 1.e-12)


def faces(n, k, stretched):
    """surface at 0, n fluid cells up to 1, k solid cells below (widths continue the grid)"""
    if not stretched:
        return torch.arange(-k, n + 1) / n
    r = 1.05 ** (32 / n)  # x1rat 1.05 at N = 32, the same map at every N
    h = (r - 1) / (r ** n - 1) * r ** torch.arange(-k, n).double()
    return torch.cat([torch.zeros(1), torch.cumsum(h, 0)]) - h[:k].sum()


def block(xf, internal=None, n2=1, scheme=0, tracer=False, nb1=0):
    """a MeshBlock; nb1 > 0: a snapy.Mesh of nb1 x1 blocks in this process (keep one alive at a time)"""
    weno = {"type": "weno5", "scale": False, "shock": False}
    eos = {"type": "ideal-gas", "gammad": GAM, "density-floor": 1.e-10, "pressure-floor": 1.e-10,
           "limiter": False}
    c = {"geometry": {"type": "cartesian",
                      "bounds": {"x1min": float(xf[0]), "x1max": float(xf[-1]), "x2min": 0., "x2max": n2 / 8.,
                                 "x3min": 0., "x3max": 1.},
                      "cells": {"nx1": len(xf) - 1, "nx2": n2, "nx3": 1, "nghost": NG}},
         "distribute": {"layout": "slab", "nb2": 1, "nb3": 1, "verbose": False},
         "dynamics": {"equation-of-state": eos, "reconstruct": {"vertical": weno, "horizontal": weno},
                      "riemann-solver": {"type": "lmars"}},
         "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                             "x2-inner": "reflecting", "x2-outer": "reflecting",
                                             "x3-inner": "periodic", "x3-outer": "periodic"}},
         "integration": {"type": "rk3", "cfl": 0.9, "implicit-scheme": scheme, "nlim": -1, "tlim": 1.e9,
                         "ncycle_out": 0},
         "forcing": {"const-gravity": {"grav1": -1.}}}
    h = xf[1:] - xf[:-1]
    if h.max() - h.min() > 1.e-12 * h.mean():  # stretched only: MPS refuses geometry/x1f
        c["geometry"]["x1f"] = [float(v) for v in xf]
    if internal:
        c["boundary-condition"]["internal"] = internal
    if tracer:
        c["scalar"] = {"nvar": 1, "names": ["q"], "reconstruct": {"type": "weno5", "shock": False}}
    if nb1:
        c["distribute"] |= {"layout": "cubed" if nb1 > 1 else "slab", "nb1": nb1, "blocks_per_process": nb1}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        yaml.safe_dump(c, open(p, "w"))
        if nb1:
            o = MeshOptions.from_yaml(p)
            o.blocks_per_process(nb1)
            m = Mesh(o)
            for b in m.blocks:
                b.to(torch.device(DEV), DT)
            return m
        o = MeshBlockOptions.from_yaml(p)
        o.layout().device(DEV)
        b = MeshBlock(o)
    b.to(torch.device(DEV), DT)
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
        out[name] = (u.clone(), wv.clone(), float((u[kIDN] * h[kk:].to(u)).sum()))
    (uw, ww, mw), (us, ws, ms) = out["wall"], out["solid"]
    return dict(diff=float((us - uw).abs().max()), wmax=float(ws[kIV1].abs().max()),
                wmax_wall=float(ww[kIV1].abs().max()), mass=abs(ms - mw) / mw,
                finite=bool(torch.isfinite(us).all()))


def thin_x2(wid, scheme, steps=20):
    """[A | full-height solid, wid cells | B] in x2 with gravity vs A and B between reflecting x2 walls:
    a column with no fluid cell has no surface (earth fill). Returns max|diff| over the fluid of A and B."""
    na, nb, n1 = 6, 5, 8
    out = {}
    for name, parts in (("solid", ((na, .3), (wid, None), (nb, .6))), ("A", ((na, .3),)), ("B", ((nb, .6),))):
        n2 = sum(c for c, _ in parts)
        b = block(torch.arange(n1 + 1.) / n1, {"solid-density": 1.e3, "solid-pressure": 1.e9}, n2, scheme)
        w = b.buffer("hydro.D").clone().zero_()
        z = (torch.arange(n1 + 2 * NG) - NG + .5) / n1
        j, d = NG, torch.zeros(n2 + 2 * NG, n1 + 2 * NG)
        solid = torch.zeros_like(w[kIDN], dtype=torch.bool)
        for c, y0 in parts:
            y = (torch.arange(c)[:, None] + .5) / c
            if y0 is None:
                solid[:, j:j + c, NG:NG + n1] = True
            else:
                d[j:j + c] = 1.e-3 * torch.exp(-((y - y0) / .2) ** 2 - ((z - .5) / .2) ** 2)
            j += c
        w[kIDN, 0] = torch.exp(-z) * (1 + d)
        w[kIPR, 0] = torch.exp(-z) * (1 + d) ** GAM
        w[kIDN][solid], w[kIPR][solid] = 1.e3, 1.e9
        init = {"hydro_w": w.to(DT)} | ({"solid": solid} if name == "solid" else {})
        var, _ = b.initialize(init)
        for _ in range(steps):
            for st in range(len(b.module("intg").stages)):
                b.forward(var, .4 / 8 / math.sqrt(GAM), st)
        out[name] = var["hydro_u"][:, 0, NG:NG + n2, NG:NG + n1]
    s = out["solid"]
    return max(float((s[:, :na] - out["A"]).abs().max()), float((s[:, na + wid:] - out["B"]).abs().max()))


def tracer(steps=20):
    """a uniform tracer (r 1e-3; 0.5 in the solid) over a step: two solid cells under half the
    columns, gravity, earth fill, a pulse; max |r / 1e-3 - 1| over the fluid"""
    n1, n2 = 8, 12
    b = block(torch.arange(n1 + 1.) / n1, {"solid-density": 1.e3, "solid-pressure": 1.e9}, n2, 0, True)
    w = b.buffer("hydro.D").clone().zero_()
    z = (torch.arange(n1 + 2 * NG) - NG + .5) / n1
    y = (torch.arange(n2 + 2 * NG) - NG + .5) / n2
    d = 1.e-3 * torch.exp(-((y[:, None] - .7) / .2) ** 2 - ((z - .5) / .2) ** 2)
    w[kIDN, 0], w[kIPR, 0] = torch.exp(-z) * (1 + d), torch.exp(-z) * (1 + d) ** GAM
    solid = torch.zeros_like(w[kIDN], dtype=torch.bool)
    solid[:, :NG + n2 // 2, :NG + 2] = True
    w[kIDN][solid], w[kIPR][solid] = 1.e3, 1.e9
    r = torch.full((1,) + tuple(solid.shape), 1.e-3)
    r[0][solid] = .5
    var, _ = b.initialize({"hydro_w": w.to(DT), "solid": solid, "scalar_r": r.to(DT)})
    for _ in range(steps):
        for st in range(3):
            b.forward(var, .4 / 8 / math.sqrt(GAM), st)
    q = var["scalar_s"][0] / var["hydro_u"][kIDN]
    fluid = ~solid[:, NG:NG + n2, NG:NG + n1]
    return float(((q[:, NG:NG + n2, NG:NG + n1] / 1.e-3 - 1).abs())[fluid].max())


def x1_split(stretched, nb1, scheme=0, n=26, k=6, steps=20):
    """the pulsed solid column (k earth cells) on nb1 x1 blocks of one snapy.Mesh: the cells over 20 steps"""
    f = faces(n, k, stretched)
    rp = fluid(f[k:], True)
    m = block(f, {"solid-density": 1.e3, "solid-pressure": 1.e9}, scheme=scheme, nb1=nb1)
    w = torch.zeros(m.blocks[0].buffer("hydro.D").size(0), 1, 1, n + k + 2 * NG)
    for c, v, s in ((kIDN, rp[0], 1.e3), (kIPR, rp[1], 1.e9)):
        w[c, 0, 0] = torch.cat([v[:1].expand(NG), torch.full((k,), s), v, v[-1:].expand(NG)])
    solid = torch.zeros_like(w[kIDN], dtype=torch.bool)
    solid[..., NG:NG + k] = True
    i0 = [int(torch.argmin((f - b.buffer("coord.x1f")[NG]).abs())) for b in m.blocks]
    sl = [slice(i, i + b.buffer("coord.x1f").numel() - 1) for i, b in zip(i0, m.blocks)]
    var, _ = m.initialize([{"hydro_w": w[..., s].clone(), "solid": solid[..., s].clone()} for s in sl])
    out = []
    for _ in range(steps):
        for st in range(3):
            m.forward(var, .4 * float((f[1:] - f[:-1]).min()) / math.sqrt(GAM), st)
        out.append(torch.cat([var[q]["hydro_u"][:, 0, 0, NG:-NG] for q in sorted(range(nb1), key=i0.__getitem__)], -1))
    return torch.stack(out)


def main():
    ok, rows = True, {}
    for stretched in (False, True) if DEV != "mps" else (False,):
        for pulse in (False, True):
            ns = (128, 256) if stretched and pulse else (32,)  # the order needs two
            for n in ns:
                for k in (1, 2, 6):
                    for fill in ("earth", "matched"):
                        r = run(n, k, stretched, pulse, fill)
                        rows[stretched, pulse, n, k, fill] = r
                        print(f"COL {'stretched' if stretched else 'uniform':9s} {'pulse' if pulse else 'rest':5s} "
                              f"N={n:3d} k={k} fill={fill:7s} max|u_solid-u_wall|={r['diff']:.2e} "
                              f"max|w| solid={r['wmax']:.2e} wall={r['wmax_wall']:.2e} "
                              f"dmass={r['mass']:.1e} finite={r['finite']}", flush=True)
                        exact = not stretched or not pulse
                        ok &= (r["finite"] and r["mass"] <= TOL / 10
                               and (not exact or r["diff"] <= TOL))
    if DEV != "mps":
        for k in (1, 2, 6):
            for fill in ("earth", "matched"):
                d = [rows[True, True, n, k, fill]["diff"] for n in (128, 256)]
                order = (math.log2(d[0] / d[1]) if d[1] > 0
                         else float("inf") if d[0] == 0 else math.nan)
                print(f"ORDER stretched pulse k={k} fill={fill}: 128->256 {order:.2f}", flush=True)
                ok &= math.isfinite(d[0]) and order >= 1.8
    for wid in (1, 2) if DEV != "mps" else ():
        for scheme in (0, 9):
            d = thin_x2(wid, scheme)
            print(f"THIN x2 full-height solid width={wid} scheme={scheme}: max|u_solid-u_wall|={d:.2e}", flush=True)
            ok &= d <= TOL
    d = tracer()
    print(f"TRACER uniform over a step, fill 0.5: max|r/r0 - 1|={d:.2e}", flush=True)
    ok &= d <= TOL
    for stretched in (False, True) if DEV != "mps" else ():  # the Mesh rows last: no MeshBlock after them
        d = (x1_split(stretched, 2) - x1_split(stretched, 1)).abs().amax((1, 2)).tolist()
        print(f"X1 SPLIT {'stretched' if stretched else 'uniform'} 2 blocks vs 1, steps 1/5/10/20: "
              + " ".join(f"{d[i - 1]:.1e}" for i in (1, 5, 10, 20)), flush=True)
        ok &= max(d) == 0.
    if DEV != "mps":
        t = time.perf_counter()
        try:
            x1_split(False, 2, scheme=9, steps=0)
            refused = "no error"
        except RuntimeError as e:
            refused = "requires nb1 = 1" in str(e) and f"refused at initialize in {time.perf_counter() - t:.1f} s"
        print(f"X1 SPLIT implicit-scheme 9 + solid, nb1 = 2: {refused}", flush=True)
        ok &= bool(refused) and refused != "no error"
    print("GATE immersed solid column:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
