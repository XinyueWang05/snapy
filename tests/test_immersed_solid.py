"""Immersed solids against walls: one table of cases per shape, each case RED on main 3fe398a9.

FLOOR: a column on k solid cells vs the wall column above it (T = 1 - z/2 balanced plus a 1e-3 pulse,
weno5 + lmars, rk3 to t = .3; N 32 uniform, 128 stretched). Every field at every step within
RT = 4096 eps of its scale (|f| for rho, p; |v| + c_s for v), final state to TOL = 1e-12, mass to TOL/10.
FILL: earth, matched, 1.1x and 2x fills give the same fluid to TOL. THIN: a full-height x2 solid
equals the columns beside it between walls. TRACER: a uniform tracer over a step stays uniform.
TERRAIN: six cubed-sphere panels over terrain on interior columns across panel edges, earth == matched.
REST: the six panels at rest over a 1-cell step: the first stage moves no fluid cell sideways (to TOL).
BALANCE: the same step started from snapy.balance_column given the deck's reference (reference=, solid=)
stays at the no-solid level for 20 steps (within 10x of the same deck without a solid).
NARROW: an x2 deck 4 and 8 columns wide with a 2-cell step beside a reflecting x2 wall, the mask given on
interior columns only: finite, and equal to the same deck with the mask mirrored into the ghosts by hand.
EDGE: the six panels over TERRAIN's mountain and ridge: every first ghost column at a panel edge carries the
neighbour's own depth, so both panels build one frame for an edge face (no phantom side wall).
EDGEREST: the six panels at rest over the mountain and ridge (interp_order 4, scheme 9): the solid deck's max |v|
over 500 steps stays within 1.3x the same deck without a solid (panel-edge ghost state not interpolated through solid).
RESTART: an x2 deck over a 2-cell step, balanced with reference='auto', solid=True, restarted from its
checkpoint: the mask comes back, so the restart resolves the same x1 wall reference, and it continues
bitwise equal to the uninterrupted run.
REFUSED at initialize: SHORT (< 3 fluid cells, < 4 with key 4), SPLIT (nb1 = 2). The sphere floor is
test_axial_momentum_flux_form.py row (n).
usage: python test_immersed_solid.py [group ...] [--device mps]   (each group in its own process;
mps: the uniform COL cases in float32 to 1e-5)"""
import math, os, subprocess, sys, tempfile
import torch, yaml
import snapy
from snapy import Mesh, MeshBlock, MeshBlockOptions, MeshOptions, kIDN, kIPR, kIV1

torch.set_default_dtype(torch.float64)
NG, GAM = 3, 1.4
DEV = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
DT, TOL = (torch.float32, 1.e-5) if DEV == "mps" else (torch.float64, 1.e-12)
RT = 4096 * torch.finfo(torch.float64).eps
FIELDS = (("rho", kIDN), ("v1", kIV1), ("v2", kIV1 + 1), ("v3", kIV1 + 2), ("p", kIPR))
EARTH = {"solid-density": 1.e3, "solid-pressure": 1.e9}
Z = (torch.arange(8 + 2 * NG) - NG + .5) / 8  # x1 centres of the 8-cell x2 decks


# a deck with a solid runs the upward x1 wall reference; its plain-wall twin asks for it explicitly
WALL_UP = {"wb-wall-reference": "upward"}
SOLID_REF = {"reference": "auto", "solid": True}  # balance_column for a solid deck


def gap(solid, wall):
    """per field: max|solid - wall| / (RT S_f), non-finite kept (pass: <= 1)"""
    cs = (GAM * wall[kIPR] / wall[kIDN]).abs().sqrt()
    return {n: float((solid[r] - wall[r]).abs().max()
                     / (RT * (wall[r].abs().max() if r in (kIDN, kIPR) else (wall[r].abs() + cs).max())))
            for n, r in FIELDS}


def worst(a, b):
    return {k: v if not math.isfinite(v) or k not in a else a[k] if not math.isfinite(a[k]) else max(a[k], v)
            for k, v in b.items()}


def faces(n, k, stretched):
    """surface at 0, n fluid cells up to 1, k solid cells below (widths continue the grid)"""
    if not stretched:
        return torch.arange(-k, n + 1) / n
    r = 1.05 ** (32 / n)  # x1rat 1.05 at N = 32, the same map at every N
    h = (r - 1) / (r ** n - 1) * r ** torch.arange(-k, n).double()
    return torch.cat([torch.zeros(1), torch.cumsum(h, 0)]) - h[:k].sum()


def block(xf, internal=None, n2=1, scheme=0, tracer=False, nb1=0, dyn=None, eos=None, extra=None):
    """a MeshBlock, or with nb1 a snapy.Mesh of nb1 x1 blocks in this process; eos replaces the
    equation of state, extra's sections are merged into the config (test_solid_refusals.py)"""
    weno = {"type": "weno5", "scale": False, "shock": False}
    eos = eos or {"type": "ideal-gas", "gammad": GAM, "density-floor": 1.e-10, "pressure-floor": 1.e-10,
                  "limiter": False}
    c = {"geometry": {"type": "cartesian",
                      "bounds": {"x1min": float(xf[0]), "x1max": float(xf[-1]), "x2min": 0., "x2max": n2 / 8.,
                                 "x3min": 0., "x3max": 1.},
                      "cells": {"nx1": len(xf) - 1, "nx2": n2, "nx3": 1, "nghost": NG}},
         "distribute": {"layout": "slab", "nb2": 1, "nb3": 1, "verbose": False},
         "dynamics": {"equation-of-state": eos, "reconstruct": {"vertical": weno, "horizontal": weno},
                      "riemann-solver": {"type": "lmars"}} | (dyn or {}),
         "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                             "x2-inner": "reflecting", "x2-outer": "reflecting",
                                             "x3-inner": "periodic", "x3-outer": "periodic"}},
         "integration": {"type": "rk3", "cfl": 0.9, "implicit-scheme": scheme, "nlim": -1, "tlim": 1.e9,
                         "ncycle_out": 0},
         "forcing": {"const-gravity": {"grav1": -1.}}}
    h = xf[1:] - xf[:-1]
    if h.max() - h.min() > 1.e-12 * h.mean():
        c["geometry"]["x1f"] = [float(v) for v in xf]
    if internal:
        c["boundary-condition"]["internal"] = internal
    for key, v in (extra or {}).items():
        c[key] = c[key] | v if isinstance(c.get(key), dict) else v
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


def fluid(f):
    """balanced cell means of the column on faces f (z above the surface), plus the pulse"""
    h = f[1:] - f[:-1]
    t = lambda z: 1. - .5 * z
    col = torch.zeros(kIPR + 1, 1, 1, len(h))
    col[kIDN, 0, 0] = (t(f[:-1]) ** 2 - t(f[1:]) ** 2) / h
    col[kIPR, 0, 0] = (t(f[:-1]) ** 3 - t(f[1:]) ** 3) / (1.5 * h)
    stretched = bool((h.max() - h.min()) > 1e-12 * h.mean())
    try:
        wb = snapy.balance_column(col, h, 1., rtol=1.e-13, max_iter=400, stretched=stretched, **SOLID_REF)[0]
    except RuntimeError:  # a fine stretched column stalls at round-off
        wb = snapy.balance_column(col, h, 1., rtol=1.e-12, max_iter=400, stretched=stretched, **SOLID_REF)[0]
    z = .5 * (f[1:] + f[:-1])
    d = 1.e-3 * torch.exp(-((z - .4) / .1) ** 2)
    wb[kIDN, 0, 0] *= 1 + d
    wb[kIPR, 0, 0] *= (1 + d) ** GAM
    return wb[[kIDN, kIPR], 0, 0]


def column(k, fill="earth", stretched=False, scheme=0, courant=.4, dyn=None):
    """floor vs wall: (per-field gap over the steps, final |u_solid - u_wall|, mass gap, final u)"""
    n = 128 if stretched else 32
    f = faces(n, k, stretched)
    rp = fluid(f[k:])
    scale = {"earth": 1., "earth1.1": 1.1, "earth2": 2.}.get(fill)
    sd, sp = (1.e3 * scale, 1.e9 * scale) if scale else (float(rp[0, 0]), float(rp[1, 0]))
    h = f[k + 1:] - f[k:-1]
    nstep = math.ceil(.3 / (courant * float(h.min()) / math.sqrt(GAM)))
    out = {}
    for name, xf, kk in (("wall", f[k:], 0), ("solid", f, k)):
        b = block(xf, {"solid-density": sd, "solid-pressure": sp} if kk else None, 1, scheme,
                  dyn=(dyn or {}) | ({} if kk else WALL_UP))
        w = b.buffer("hydro.D").clone().zero_()
        for c, v in ((kIDN, rp[0]), (kIPR, rp[1])):
            w[c, 0, 0, NG + kk:NG + kk + n] = v
            w[c, 0, 0, :NG + kk] = v[0]
            w[c, 0, 0, NG + kk + n:] = v[-1]
        init = {"hydro_w": w}
        if kk:
            init["solid"] = torch.zeros_like(w[kIDN], dtype=torch.bool)
            init["solid"][..., NG:NG + kk] = True
            w[kIDN][init["solid"]], w[kIPR][init["solid"]] = sd, sp
        var, _ = b.initialize({k: v.to(DEV) for k, v in init.items()})
        out[name] = []
        for _ in range(nstep):
            for st in range(3):
                b.forward(var, .3 / nstep, st)
            out[name].append(var["hydro_w"][:5, 0, 0, NG + kk:NG + kk + n].cpu().double())
        out[name + "_u"] = var["hydro_u"][:, 0, 0, NG + kk:NG + kk + n].cpu().double()
    g = {}
    for s, w in zip(out["solid"], out["wall"]):
        g = worst(g, gap(s, w))
    us, uw = out["solid_u"], out["wall_u"]
    mass = abs(float(((us[kIDN] - uw[kIDN]) * h).sum() / (uw[kIDN] * h).sum()))
    return g, float((us - uw).abs().max()), mass, us


def floor_row(tag, *args, **kw):
    try:
        g, d, m, _ = column(*args, **kw)
    except RuntimeError as e:
        print(f"{tag}: RAISED {str(e).splitlines()[0][:200]} FAIL", flush=True)
        return False
    ok = (DEV == "mps" or all(math.isfinite(v) and v <= 1. for v in g.values())) and d <= TOL \
        and (DEV == "mps" or m <= TOL / 10)
    print(f"{tag}: gap/(RT S_f) " + " ".join(f"{k} {v:.2e}" for k, v in g.items())
          + f" max|u_solid-u_wall| {d:.2e} dmass {m:.1e} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def refused(tag, fn, needle):
    try:
        fn()
        msg = "no error"
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
    ok = needle in msg
    print(f"{tag}: {'refused' if ok else 'NOT refused (' + msg[:150] + ')'} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def short(n, k, dyn=None):
    """a column of n fluid cells on k earth cells: initialize only"""
    b = block(faces(n, k, False), EARTH, dyn=dyn)
    w = b.buffer("hydro.D").clone().zero_()
    w[kIDN], w[kIPR] = 1., 1.
    solid = torch.zeros_like(w[kIDN], dtype=torch.bool)
    solid[..., NG:NG + k] = True
    b.initialize({"hydro_w": w, "solid": solid})


def split(scheme, n=26, k=6):
    """the solid column (k earth cells) on two x1 blocks of one snapy.Mesh: initialize only"""
    f = faces(n, k, False)
    m = block(f, EARTH, scheme=scheme, nb1=2)
    w = torch.ones(m.blocks[0].buffer("hydro.D").size(0), 1, 1, n + k + 2 * NG)
    solid = torch.zeros_like(w[kIDN], dtype=torch.bool)
    solid[..., NG:NG + k] = True
    i0 = [int(torch.argmin((f - b.buffer("coord.x1f")[NG].cpu()).abs())) for b in m.blocks]
    sl = [slice(i, i + b.buffer("coord.x1f").numel() - 1) for i, b in zip(i0, m.blocks)]
    m.initialize([{"hydro_w": w[..., s].clone(), "solid": solid[..., s].clone()} for s in sl])


def x2_deck(n2, d, solid, scheme=0, tracer=False, dyn=None):
    """8 x1 cells, n2 wide in x2, earth fill: the state exp(-z) (1 + d), solid cells filled"""
    b = block(torch.arange(9.) / 8, EARTH, n2, scheme, tracer, dyn=dyn)
    w = b.buffer("hydro.D").clone().zero_()
    w[kIDN, 0], w[kIPR, 0] = torch.exp(-Z) * (1 + d), torch.exp(-Z) * (1 + d) ** GAM
    w[kIDN][solid], w[kIPR][solid] = 1.e3, 1.e9
    return b, w


def steps(b, var, n=20):
    for _ in range(n):
        for st in range(3):
            b.forward(var, .4 / 8 / math.sqrt(GAM), st)


def thin_x2(wid, scheme):
    """[A | full-height solid, wid cells | B] in x2 vs A and B between reflecting x2 walls: max|diff|"""
    na, nb, out = 6, 5, {}
    for name, parts in (("solid", ((na, .3), (wid, None), (nb, .6))), ("A", ((na, .3),)), ("B", ((nb, .6),))):
        n2 = sum(c for c, _ in parts)
        j, d = NG, torch.zeros(n2 + 2 * NG, 8 + 2 * NG)
        solid = torch.zeros(1, n2 + 2 * NG, 8 + 2 * NG, dtype=torch.bool)
        for c, y0 in parts:
            if y0 is None:
                solid[:, j:j + c, NG:NG + 8] = True
            else:
                y = (torch.arange(c)[:, None] + .5) / c
                d[j:j + c] = 1.e-3 * torch.exp(-((y - y0) / .2) ** 2 - ((Z - .5) / .2) ** 2)
            j += c
        b, w = x2_deck(n2, d, solid, scheme, dyn=None if name == "solid" else WALL_UP)
        var, _ = b.initialize({"hydro_w": w} | ({"solid": solid} if name == "solid" else {}))
        steps(b, var)
        out[name] = var["hydro_u"][:, 0, NG:NG + n2, NG:NG + 8]
    s = out["solid"]
    return max(float((s[:, :na] - out["A"]).abs().max()), float((s[:, na + wid:] - out["B"]).abs().max()))


def terrain(scheme, nstep=10):
    """the six-panel deck of test_axial_momentum_flux_form at rest over terrain given on interior
    columns only (ghost depths are the API's): a 3-cell mountain centred on a panel edge and a 2-cell
    equatorial ridge across the panel edges; earth vs matched fill, the earth run's dt on both"""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import test_axial_momentum_flux_form as afm
    cfg = yaml.safe_load(open(os.path.join(here, "test_axial_momentum_flux_form.yaml")))
    cfg["integration"]["implicit-scheme"] = scheme
    mesh, blocks = afm.build(cfg, "cpu")
    la0, lo0 = (blocks[0].module("coord").buffer(n)[..., 0] for n in ("latitude", "longitude"))
    lon0, lat0 = (float(x[NG + 4, -NG - 1] + x[NG + 4, -NG]) / 2 for x in (lo0, la0))  # panel 0's x2 edge
    dlat = float((la0[NG + 1, NG + 4] - la0[NG, NG + 4]).abs())
    w0 = [v["hydro_w"].clone() for v in afm.initial_state(cfg, blocks, None, False, solid=True)]
    masks = []
    for b in blocks:
        lon, lat = (b.module("coord").buffer(n)[..., 0] for n in ("longitude", "latitude"))
        d = torch.arccos((lat.sin() * math.sin(lat0) + lat.cos() * math.cos(lat0) * (lon - lon0).cos()).clamp(-1, 1))
        h = torch.maximum(torch.round(3. * torch.exp(-(d / (1.5 * dlat)) ** 2)).long(),
                          torch.where(lat.abs() < .6 * dlat, 2, 0))
        s, at = torch.zeros(w0[0].shape[1:], dtype=torch.bool), torch.arange(w0[0].size(-1))
        s[NG:-NG, NG:-NG] = (at >= NG) & (at < NG + h[NG:-NG, NG:-NG, None])  # interior columns only
        masks.append(s)
    del mesh, blocks
    out, dts, fin = {}, [], True
    for fill in ("earth", "matched"):
        c = yaml.safe_load(yaml.safe_dump(cfg))
        c["boundary-condition"]["internal"] = EARTH if fill == "earth" else {
            "solid-density": float(w0[0][kIDN, NG, NG, NG]), "solid-pressure": float(w0[0][kIPR, NG, NG, NG])}
        mesh, blocks = afm.build(c, "cpu")
        mv, _ = mesh.initialize([{"hydro_w": w.clone(), "solid": s.clone()} for w, s in zip(w0, masks)])
        out[fill] = []
        for cycle in range(1, nstep + 1):
            mesh.set_cycle(cycle)
            if fill == "earth":
                dts.append(mesh.max_time_step(mv))
            for st in range(len(blocks[0].module("intg").stages)):
                mesh.forward(mv, dts[cycle - 1], st)
            fl = [(v["hydro_w"][:5], ~v["solid"]) for v in mv]  # ghost columns: the API's depth
            fin &= all(bool(torch.isfinite(x[:, f]).all()) for x, f in fl)
            out[fill].append([x[:, f].double() for x, f in fl])
        del mesh, blocks, mv
    d = max(float((a - b).abs().max()) for x, y in zip(out["earth"], out["matched"]) for a, b in zip(x, y))
    ok = d == 0 and fin
    print(f"TERRAIN cubed sphere, mountain on a panel edge + ridge, scheme {scheme}: max|earth - matched| "
          f"{d:.2e} finite {fin} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def rest_step(scheme):
    """test_axial_momentum_flux_form's deck at rest, one earth cell under every column north of the
    equator: max |d(rho v2, rho v3)| / rho over the fluid after the first stage (the side-wall p*)"""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import test_axial_momentum_flux_form as afm
    cfg = yaml.safe_load(open(os.path.join(here, "test_axial_momentum_flux_form.yaml")))
    cfg["integration"]["implicit-scheme"] = scheme
    cfg["boundary-condition"]["internal"] = EARTH
    mesh, blocks = afm.build(cfg, "cpu")
    w0 = [v["hydro_w"].clone() for v in afm.initial_state(cfg, blocks, None, False, solid=True)]
    masks, at = [], torch.arange(w0[0].size(-1))
    for b in blocks:
        h = (b.module("coord").buffer("latitude")[..., 0] > 0).long()
        s = torch.zeros(w0[0].shape[1:], dtype=torch.bool)
        s[NG:-NG, NG:-NG] = (at >= NG) & (at < NG + h[NG:-NG, NG:-NG, None])
        masks.append(s)
    mv, _ = mesh.initialize([{"hydro_w": w.clone(), "solid": s} for w, s in zip(w0, masks)])
    u0 = [v["hydro_u"].clone() for v in mv]
    mesh.set_cycle(1)
    mesh.forward(mv, mesh.max_time_step(mv), 0)
    d = 0.
    for v, a in zip(mv, u0):
        f = (~v["solid"])[NG:-NG, NG:-NG, NG:-NG]
        q = ((v["hydro_u"] - a)[kIV1 + 1:kIV1 + 3].abs().amax(0) / a[kIDN])[NG:-NG, NG:-NG, NG:-NG]
        d = max(d, float(q[f].max()))
    print(f"REST cubed sphere over a 1-cell step, scheme {scheme}: first stage max|d(rho v_h)|/rho {d:.2e} "
          f"{'PASS' if d <= TOL else 'FAIL'}", flush=True)
    return d <= TOL


def rest_balanced(scheme, solid, kw, nstep=20):
    """the six panels at rest, the start an isothermal column through snapy.balance_column(**kw), one earth
    cell under every column north of the equator if `solid`: max |v| over the fluid in nstep steps"""
    import snapy
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import test_axial_momentum_flux_form as afm
    cfg = yaml.safe_load(open(os.path.join(here, "test_axial_momentum_flux_form.yaml")))
    cfg["integration"]["implicit-scheme"] = scheme
    if solid:
        cfg["boundary-condition"]["internal"] = EARTH
    mesh, blocks = afm.build(cfg, "cpu")
    g, n1 = -cfg["forcing"]["const-gravity"]["grav1"], cfg["geometry"]["cells"]["nx1"]
    init = []
    for b in blocks:
        c = b.module("coord")
        r, dr = c.buffer("x1v")[NG:NG + n1].cpu(), c.buffer("dx1f")[NG:NG + n1].cpu().contiguous()
        col = torch.zeros(kIPR + 1, 1, 1, n1)
        col[kIPR, 0, 0] = afm.P0 * torch.exp(-(r - r[0]) * afm.RHO0 * g / afm.P0)
        col[kIDN, 0, 0] = col[kIPR, 0, 0] * afm.RHO0 / afm.P0
        wb = snapy.balance_column(col, dr, g, rtol=1.e-13, **kw)[0]
        w = b.buffer("hydro.D").clone().zero_()
        for q in (kIDN, kIPR):
            w[q][..., NG:NG + n1] = wb[q, 0, 0]
            w[q][..., :NG], w[q][..., NG + n1:] = w[q][..., NG:NG + 1], w[q][..., NG + n1 - 1:NG + n1]
        s = torch.zeros(w.shape[1:], dtype=torch.bool)
        if solid:
            h, at = (c.buffer("latitude")[..., 0] > 0).long(), torch.arange(w.size(-1))
            s[NG:-NG, NG:-NG] = (at >= NG) & (at < NG + h[NG:-NG, NG:-NG, None])
        init.append({"hydro_w": w} | ({"solid": s} if solid else {}))
    mv, _ = mesh.initialize(init)
    vmax = 0.
    for n in range(1, nstep + 1):
        mesh.set_cycle(n)
        dt = mesh.max_time_step(mv)
        for st in range(len(blocks[0].module("intg").stages)):
            mesh.forward(mv, dt, st)
        for v in mv:
            f = (~v["solid"] if solid else torch.ones_like(v["hydro_w"][0], dtype=torch.bool))[NG:-NG, NG:-NG, NG:-NG]
            vmax = max(vmax, float((v["hydro_w"][kIV1:kIV1 + 3, NG:-NG, NG:-NG, NG:-NG].abs() * f).max()))
    return vmax


def balance(scheme):
    """BALANCE: the deck's reference given to balance_column holds the no-solid level; 10x covers the
    2000-step ratio of the upward-balanced start (4.3e-11 / 1.7e-10 against 2.2e-11)"""
    ref = rest_balanced(scheme, False, {})
    d = rest_balanced(scheme, True, SOLID_REF)
    ok = d <= 10. * ref
    print(f"BALANCE cubed sphere over a 1-cell step, scheme {scheme}: 20-step max|v| {d:.2e} "
          f"(no solid {ref:.2e}, ratio {d / ref:.2f}) {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def edge():
    """EDGE: first-layer panel-edge ghost columns whose depth differs from the nearest neighbour column's"""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import test_axial_momentum_flux_form as afm
    cfg = yaml.safe_load(open(os.path.join(here, "test_axial_momentum_flux_form.yaml")))
    cfg["boundary-condition"]["internal"] = EARTH
    mesh, blocks = afm.build(cfg, "cpu")
    la0, lo0 = (blocks[0].module("coord").buffer(n)[..., 0] for n in ("latitude", "longitude"))
    lon0, lat0 = (float(x[NG + 4, -NG - 1] + x[NG + 4, -NG]) / 2 for x in (lo0, la0))
    dlat = float((la0[NG + 1, NG + 4] - la0[NG, NG + 4]).abs())
    w0 = [v["hydro_w"].clone() for v in afm.initial_state(cfg, blocks, None, False, solid=True)]
    init, xyz = [], []
    for b, w in zip(blocks, w0):
        lon, lat = (b.module("coord").buffer(n)[..., 0] for n in ("longitude", "latitude"))
        d = torch.arccos((lat.sin() * math.sin(lat0) + lat.cos() * math.cos(lat0) * (lon - lon0).cos()).clamp(-1, 1))
        h = torch.maximum(torch.round(3. * torch.exp(-(d / (1.5 * dlat)) ** 2)).long(),
                          torch.where(lat.abs() < .6 * dlat, 2, 0))
        s, at = torch.zeros(w.shape[1:], dtype=torch.bool), torch.arange(w.size(-1))
        s[NG:-NG, NG:-NG] = (at >= NG) & (at < NG + h[NG:-NG, NG:-NG, None])
        init.append({"hydro_w": w, "solid": s})
        xyz.append(torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], -1))
    mv, _ = mesh.initialize(init)
    n1 = cfg["geometry"]["cells"]["nx1"]
    dep = [v["solid"][..., NG:NG + n1].long().cumprod(-1).sum(-1) for v in mv]
    nx, ng = dep[0].size(0) - 2 * NG, NG
    cols = [(i, k, j) for i in range(6) for k in range(ng, ng + nx) for j in range(ng, ng + nx)]
    pts = torch.stack([xyz[i][k, j] for i, k, j in cols])
    bad = 0
    for i in range(6):
        for m in range(ng, ng + nx):
            for k, j in ((m, ng - 1), (m, ng + nx), (ng - 1, m), (ng + nx, m)):
                order = torch.argsort((pts - xyz[i][k, j]).norm(dim=-1))
                o = next(int(q) for q in order if cols[int(q)][0] != i)
                bad += int(dep[i][k, j]) != int(dep[cols[o][0]][cols[o][1], cols[o][2]])
    print(f"EDGE six panels over the mountain and ridge: first-layer edge ghost columns off the neighbour's depth "
          f"{bad} {'PASS' if bad == 0 else 'FAIL'}", flush=True)
    return bad == 0


def edge_rest(scheme=9, nstep=500):
    """EDGEREST: max |v| over the fluid in nstep steps, solid deck vs the same deck without a solid"""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import test_axial_momentum_flux_form as afm
    vm = {}
    for solid in (True, False):
        cfg = yaml.safe_load(open(os.path.join(here, "test_axial_momentum_flux_form.yaml")))
        cfg["integration"]["implicit-scheme"] = scheme
        cfg["geometry"]["cells"]["interp_order"] = 4
        if solid:
            cfg["boundary-condition"]["internal"] = EARTH
        mesh, blocks = afm.build(cfg, "cpu")
        la0, lo0 = (blocks[0].module("coord").buffer(n)[..., 0] for n in ("latitude", "longitude"))
        lon0, lat0 = (float(x[NG + 4, -NG - 1] + x[NG + 4, -NG]) / 2 for x in (lo0, la0))
        dlat = float((la0[NG + 1, NG + 4] - la0[NG, NG + 4]).abs())
        w0 = [v["hydro_w"].clone() for v in afm.initial_state(cfg, blocks, None, False, solid=solid)]
        init = []
        for b, w in zip(blocks, w0):
            lon, lat = (b.module("coord").buffer(n)[..., 0] for n in ("longitude", "latitude"))
            d = torch.arccos((lat.sin() * math.sin(lat0) + lat.cos() * math.cos(lat0) * (lon - lon0).cos()).clamp(-1, 1))
            h = torch.maximum(torch.round(3. * torch.exp(-(d / (1.5 * dlat)) ** 2)).long(),
                              torch.where(lat.abs() < .6 * dlat, 2, 0))
            s, at = torch.zeros(w.shape[1:], dtype=torch.bool), torch.arange(w.size(-1))
            s[NG:-NG, NG:-NG] = (at >= NG) & (at < NG + h[NG:-NG, NG:-NG, None])
            init.append({"hydro_w": w} | ({"solid": s} if solid else {}))
        mv, _ = mesh.initialize(init)
        m = 0.
        for n in range(1, nstep + 1):
            mesh.set_cycle(n)
            dt = mesh.max_time_step(mv)
            for st in range(len(blocks[0].module("intg").stages)):
                mesh.forward(mv, dt, st)
            for v in mv:
                f = ~v["solid"] if solid else torch.ones_like(v["hydro_w"][0], dtype=torch.bool)
                m = max(m, float((v["hydro_w"][kIV1:kIV1 + 3].abs().amax(0) * f)[NG:-NG, NG:-NG, NG:-NG].max()))
        vm[solid] = m
    r = vm[True] / vm[False]
    print(f"EDGEREST six panels at rest over the mountain and ridge, scheme {scheme}, {nstep} steps: max|v| solid "
          f"{vm[True]:.3e} no solid {vm[False]:.3e} ratio {r:.2f} {'PASS' if r <= 1.3 else 'FAIL'}", flush=True)
    return r <= 1.3


def narrow(nx, scheme=0, nstep=5):
    """NARROW: the step deck nx columns wide, interior mask vs mirrored ghosts: (finite, max|diff|)"""
    k = 2
    f = faces(32, k, False)
    rp = fluid(f)
    out = []
    for mirrored in (False, True):
        b = block(f, EARTH, nx, scheme)
        w = b.buffer("hydro.D").clone().zero_()
        for c, v in ((kIDN, rp[0]), (kIPR, rp[1])):
            w[c, 0, :, NG:-NG] = v
            w[c, 0, :, :NG], w[c, 0, :, -NG:] = v[0], v[-1]
        solid = torch.zeros_like(w[kIDN], dtype=torch.bool)
        solid[:, NG:NG + nx // 2, NG:NG + k] = True
        if mirrored:  # reflecting x2 walls: the ghosts mirror the interior
            solid[:, :NG], solid[:, -NG:] = solid[:, NG:2 * NG].flip(1), solid[:, -2 * NG:-NG].flip(1)
        w[kIDN][solid], w[kIPR][solid] = EARTH["solid-density"], EARTH["solid-pressure"]
        var, _ = b.initialize({"hydro_w": w.to(DEV), "solid": solid.to(DEV)})
        for _ in range(nstep):
            for st in range(3):
                b.forward(var, .3 / 40, st)
        out.append(var["hydro_u"][:, :, NG:-NG, NG:-NG].cpu().double())
    fin = bool(torch.isfinite(out[0]).all())
    d = float((out[0] - out[1]).abs().max()) if fin else float("nan")
    ok = fin and d == 0.
    print(f"NARROW x2 step beside a reflecting wall, {nx} columns: finite {fin}, max|interior mask - mirrored| "
          f"{d:.2e} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def restart(scheme, n1=5, n2=5):
    """RESTART: max |u_restarted - u_uninterrupted| after n2 steps (0 = bitwise), the mask restored"""
    k, nx = 2, 8
    f = faces(32, k, False)
    rp = fluid(f)  # the whole column, balanced for this solid deck (SOLID_REF)
    dt = .3 / 40
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            def deck():
                return block(f, EARTH, nx, scheme, extra={"outputs": [{"type": "restart", "dt": 1.e9}]})
            b = deck()
            w = b.buffer("hydro.D").clone().zero_()
            for c, v in ((kIDN, rp[0]), (kIPR, rp[1])):
                w[c, 0, :, NG:-NG] = v
                w[c, 0, :, :NG], w[c, 0, :, -NG:] = v[0], v[-1]
            solid = torch.zeros_like(w[kIDN], dtype=torch.bool)
            solid[:, :NG + nx // 2, NG:NG + k] = True  # a step: half the columns stand on 2 earth cells
            w[kIDN][solid], w[kIPR][solid] = EARTH["solid-density"], EARTH["solid-pressure"]
            var, _ = b.initialize({"hydro_w": w.to(DEV), "solid": solid.to(DEV)})
            run = lambda blk, v, n: [blk.forward(v, dt, st) for _ in range(n) for st in range(3)]
            run(b, var, n1)
            b.make_outputs(var, n1 * dt, True)
            path = [x for x in os.listdir(".") if x.endswith(".final.restart")][0]
            run(b, var, n2)
            r = deck()
            rv, _ = r.initialize_from_restart(path)
            back = "solid" in rv and torch.equal(rv["solid"].cpu(), solid)
            run(r, rv, n2)
            gap = float((rv["hydro_u"] - var["hydro_u"]).abs().max())
        finally:
            os.chdir(cwd)
    ok = back and gap == 0.
    print(f"RESTART x2 step, scheme {scheme}: mask restored {back}, max|u_restart - u_run| after {n2} steps "
          f"{gap:.2e} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def tracer():
    """a uniform tracer (r 1e-3; 0.5 in the solid) over a two-cell step: max |r / 1e-3 - 1| over the fluid"""
    n2 = 12
    y = (torch.arange(n2 + 2 * NG) - NG + .5) / n2
    solid = torch.zeros(1, n2 + 2 * NG, 8 + 2 * NG, dtype=torch.bool)
    solid[:, :NG + n2 // 2, :NG + 2] = True
    b, w = x2_deck(n2, 1.e-3 * torch.exp(-((y[:, None] - .7) / .2) ** 2 - ((Z - .5) / .2) ** 2), solid,
                   tracer=True)
    r = torch.full((1,) + tuple(solid.shape), 1.e-3)
    r[0][solid] = .5
    var, _ = b.initialize({"hydro_w": w, "solid": solid, "scalar_r": r})
    steps(b, var)
    q = var["scalar_s"][0] / var["hydro_u"][kIDN]
    return float(((q[:, NG:NG + n2, NG:NG + 8] / 1.e-3 - 1).abs())[~solid[:, NG:NG + n2, NG:NG + 8]].max())


# floor-vs-wall cases: group, column() keywords (k, fill, stretched, scheme, courant, dyn)
FLOOR = [("col", dict(k=k, stretched=st)) for st in (False, True) for k in (1, 6)] + [
    ("impl", dict(k=k, fill=fl, scheme=sc, courant=cr))
    for sc, cr, k, fl in ((1, 4., 6, "earth"), (9, 4., 6, "earth"), (9, .4, 1, "earth"), (9, 4., 6, "matched"))] + [
    (g, dict(k=k, dyn=dyn)) for g, dyn in (("cov", {"x1-mass-covariance": True}), ("conv4", {"conversion-order": 4}))
    for k in (1, 6)]
FILL = [dict(k=2, fill=f, stretched=st) for st in (False, True) for f in ("matched", "earth1.1", "earth2")]
THIN = ((1, 0), (1, 9), (2, 0))  # (width, scheme)
REFUSED = {"short": (("SHORT k=30 solid, 2 fluid cells", lambda: short(2, 30), "need at least 3"),
                     ("SHORT conversion-order 4, 3 fluid cells", lambda: short(3, 30, {"conversion-order": 4}),
                      "needs at least 4")),
           "split": (("SPLIT solid surface, nb1 = 2", lambda: split(0), "whole x1 column"),  # Mesh: own process
                     ("SPLIT implicit-scheme 9 + solid, nb1 = 2", lambda: split(9), "requires nb1 = 1"))}


def group(name):
    if DEV == "mps":  # float32: the uniform floor rows only
        return name != "col" or all([floor_row(f"COL uniform k={k} (mps)", k) for k in (1, 6)])
    if name in REFUSED:
        return all([refused(*case) for case in REFUSED[name]])
    if name == "thin":
        d = [thin_x2(wid, sc) for wid, sc in THIN]
        for (wid, sc), v in zip(THIN, d):
            print(f"THIN x2 full-height solid width={wid} scheme={sc}: max|u_solid-u_wall|={v:.2e} "
                  f"{'PASS' if v <= TOL else 'FAIL'}", flush=True)
        return all(v <= TOL for v in d)
    if name == "terrain":
        return all([terrain(scheme) for scheme in (0, 9)])
    if name == "edgerest":
        return edge_rest()
    if name == "edge":
        return edge()
    if name == "narrow":
        return all([narrow(nx) for nx in (4, 8)])
    if name == "restart":
        return all([restart(scheme) for scheme in (0, 9)])
    if name == "balance":
        return all([balance(scheme) for scheme in (0, 9)])
    if name == "rest":
        return all([rest_step(scheme) for scheme in (0, 9)])
    if name == "tracer":
        d = tracer()
        print(f"TRACER uniform over a step, fill 0.5: max|r/r0 - 1|={d:.2e} {'PASS' if d <= TOL else 'FAIL'}",
              flush=True)
        return d <= TOL
    if name == "fill":  # each fill against the earth fill of the same column
        earth = {st: column(2, stretched=st)[3] for st in (False, True)}
        d = [float((column(**kw)[3] - earth[kw["stretched"]]).abs().max()) for kw in FILL]
        for kw, v in zip(FILL, d):
            print(f"FILL {kw} vs earth: max|u - u_earth|={v:.2e} {'PASS' if v <= TOL else 'FAIL'}", flush=True)
        return all(v <= TOL for v in d)
    return all([floor_row(f"{g.upper()} " + " ".join(f"{a}={b}" for a, b in kw.items()), **kw)
                for g, kw in FLOOR if g == name])


GROUPS = ("col", "fill", "impl", "cov", "conv4", "thin", "tracer", "terrain", "rest", "balance",
          "restart", "narrow", "edge", "edgerest", "short", "split")


def main():
    rows = [a for a in sys.argv[1:] if a in GROUPS] or list(GROUPS)
    if len(rows) > 1:
        dev = ["--device", DEV] if DEV != "cpu" else []
        ok = all([subprocess.run([sys.executable, os.path.abspath(__file__), r] + dev).returncode == 0
                  for r in rows])
    else:
        ok = bool(group(rows[0]))
    print(f"GATE immersed solid {' '.join(rows)}:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
