"""Immersed-solid gates: a column on a bottom-attached solid equals the plain wall column above it
to round-off in EVERY primitive field (rho, v1, v2, v3, p), at every step.

Round-off tolerance, per field f and step: max|f_solid - f_wall| over the fluid cells
<= RT * S_f, RT = 4096 eps (9.1e-13), S_f the wall's max |f| for rho and p and, for the three
velocity components, the wall's max (|v| + c_s), c_s = sqrt(GAMMA p / rho) (velocity round-off
is set by the acoustic scale, not by |v|, which is zero in a column at rest).

COV:  the uniform pulse column (T = 1 - z/2, p = T^2, rho = T, g = 1, N 32, weno5 + lmars, rk3,
      balanced start plus a 1e-3 pulse, to t = 0.3) with dynamics/x1-mass-covariance on; solid
      depth k = 1, 2, 6 cells; fill earth (1e3, 1e9) or matched (the first fluid cell's state).
CONV4: the same column with dynamics/conversion-order 4, k = 1, 2, 6, fill earth and matched.
SPHERE: the cubed-sphere deck of test_axial_momentum_flux_form.py (full, interp_order 4,
      closure off, Coriolis off, perturbed jet, balanced), a uniform 2-cell solid floor (default
      fill) vs the wall two cells up, the floor's initial fluid state and dt on both, NSTEP steps.

TERRAIN: the same deck at rest over terrain given on interior columns only, a mountain centred on a
      panel edge and an equatorial ridge across the panel edges (a reconstruction of Zoey's arm);
      earth == matched fill bitwise and finite, explicit and implicit-scheme 9.

usage: python test_solid_gates.py [cov] [conv4] [sphere] [terrain] [cpu] [cuda]   (no argument: all rows, cpu)"""
import math, os, subprocess, sys, tempfile
import torch, yaml
import snapy
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

torch.set_default_dtype(torch.float64)
NG, GAM = 3, 1.4
RT = 4096 * torch.finfo(torch.float64).eps
NSTEP = 20  # SPHERE
FIELDS = (("rho", kIDN), ("v1", kIV1), ("v2", kIV1 + 1), ("v3", kIV1 + 2), ("p", kIPR))


def gap(solid, wall, gamma=GAM):
    """per field: max|solid - wall| / (RT * S_f) over prims [5, ...] (pass: <= 1)"""
    cs = (gamma * wall[kIPR] / wall[kIDN]).abs().sqrt()
    out = {}
    for name, r in FIELDS:
        s = wall[r].abs().max() if r in (kIDN, kIPR) else (wall[r].abs() + cs).max()
        out[name] = float((solid[r] - wall[r]).abs().max() / (RT * s))
    return out


def worst(a, b):
    """per-field max over steps; a non-finite gap (a NaN state) is kept, as max() would drop it"""
    out = {}
    for k, v in b.items():
        u = a.get(k, 0.)
        out[k] = u if not math.isfinite(u) else v if not math.isfinite(v) else max(u, v)
    return out


def report(row, g):
    ok = all(v <= 1. for v in g.values()) and all(math.isfinite(v) for v in g.values())
    print(f"{row}: max|diff| / (RT S_f) " + " ".join(f"{k} {v:.2e}" for k, v in g.items())
          + (" PASS" if ok else " FAIL"), flush=True)
    return ok


# ---------------- COV / CONV4: one column, MeshBlock ----------------
def block(xf, internal=None, dyn=None):
    weno = {"type": "weno5", "scale": False, "shock": False}
    eos = {"type": "ideal-gas", "gammad": GAM, "density-floor": 1.e-10, "pressure-floor": 1.e-10,
           "limiter": False}
    c = {"geometry": {"type": "cartesian",
                      "bounds": {"x1min": float(xf[0]), "x1max": float(xf[-1]), "x2min": 0., "x2max": 1. / 8.,
                                 "x3min": 0., "x3max": 1.},
                      "cells": {"nx1": len(xf) - 1, "nx2": 1, "nx3": 1, "nghost": NG}},
         "distribute": {"layout": "slab", "nb2": 1, "nb3": 1, "verbose": False},
         "dynamics": {"equation-of-state": eos, "reconstruct": {"vertical": weno, "horizontal": weno},
                      "riemann-solver": {"type": "lmars"}} | (dyn or {}),
         "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                             "x2-inner": "reflecting", "x2-outer": "reflecting",
                                             "x3-inner": "periodic", "x3-outer": "periodic"}},
         "integration": {"type": "rk3", "cfl": 0.9, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9,
                         "ncycle_out": 0},
         "forcing": {"const-gravity": {"grav1": -1.}}}
    if internal:
        c["boundary-condition"]["internal"] = internal
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        yaml.safe_dump(c, open(p, "w"))
        o = MeshBlockOptions.from_yaml(p)
        o.layout().device("cpu")
        b = MeshBlock(o)
    b.to(torch.device("cpu"), torch.float64)
    return b


def fluid(f):
    """balanced cell means of the column on faces f (z above the surface), plus the pulse"""
    h = f[1:] - f[:-1]
    t = lambda z: 1. - .5 * z
    col = torch.zeros(kIPR + 1, 1, 1, len(h))
    col[kIDN, 0, 0] = (t(f[:-1]) ** 2 - t(f[1:]) ** 2) / h
    col[kIPR, 0, 0] = (t(f[:-1]) ** 3 - t(f[1:]) ** 3) / (1.5 * h)
    wb = snapy.balance_column(col, h, 1., rtol=1.e-13, max_iter=400, geometry="cartesian")[0]
    z = .5 * (f[1:] + f[:-1])
    d = 1.e-3 * torch.exp(-((z - .4) / .1) ** 2)
    wb[kIDN, 0, 0] *= 1 + d
    wb[kIPR, 0, 0] *= (1 + d) ** GAM
    return wb[[kIDN, kIPR], 0, 0]


def column(k, fill, dyn, n=32, courant=.4):
    """floor (k solid cells under n fluid cells) vs wall (the n fluid cells), the wall's dt, to t = .3"""
    f = torch.arange(-k, n + 1) / n
    rp = fluid(f[k:])
    sd, sp = (1.e3, 1.e9) if fill == "earth" else (float(rp[0, 0]), float(rp[1, 0]))
    nstep = math.ceil(.3 / (courant * float(f[1] - f[0]) / math.sqrt(GAM)))
    dt, steps = .3 / nstep, {}
    for name, xf, kk in (("wall", f[k:], 0), ("solid", f, k)):
        b = block(xf, {"solid-density": sd, "solid-pressure": sp} if kk else None, dyn)
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
        steps[name] = []
        for _ in range(nstep):
            for st in range(3):
                b.forward(var, dt, st)
            steps[name].append(var["hydro_w"][:5, 0, 0, NG + kk:NG + kk + n].clone())
    g = {}
    for s, w in zip(steps["solid"], steps["wall"]):
        g = worst(g, gap(s, w))
    return g


def row_column(tag, dyn):
    ok = True
    for k in (1, 2, 6):
        for fill in ("earth", "matched"):
            try:
                ok &= report(f"{tag} uniform pulse N=32 k={k} fill={fill}", column(k, fill, dyn))
            except RuntimeError as e:
                print(f"{tag} uniform pulse N=32 k={k} fill={fill}: RAISED {str(e).splitlines()[0][:300]} FAIL",
                      flush=True)
                ok = False
    return ok


# ---------------- SPHERE: six cubed-sphere panels, Mesh ----------------
def sphere():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_axial_momentum_flux_form as afm
    base = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "test_axial_momentum_flux_form.yaml")))
    k = 2

    def config(kk):
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["dynamics"]["conserve-axial-momentum"] = "full"
        cfg["dynamics"]["conserve-axial-momentum-closure"] = False
        cfg["geometry"]["cells"]["interp_order"] = 4
        b = cfg["geometry"]["bounds"]
        n = cfg["geometry"]["cells"]["nx1"]
        b["x1min"] = float(b["x1min"]) + kk * (float(b["x1max"]) - float(b["x1min"])) / n
        cfg["geometry"]["cells"]["nx1"] = n - kk
        return cfg

    def jet(lon, lat, rc):
        return afm.W_SB * rc + afm.U_PERT * torch.cos(lat) ** 2 * torch.sin(2.0 * lon)

    mesh, blocks = afm.build(config(0), "cpu")
    w0 = [v["hydro_w"].clone() for v in afm.initial_state(config(0), blocks, jet, True)]
    del mesh, blocks
    dts, out = [], {}
    for name, kk in (("floor", 0), ("wall", k)):
        mesh, blocks = afm.build(config(kk), "cpu")
        mv = [{"hydro_w": w[..., kk:].clone()} for w in w0]  # the wall deck drops k rows below
        if kk == 0:
            for v in mv:
                v["solid"] = torch.zeros(v["hydro_w"].shape[1:], dtype=torch.bool)
                v["solid"][..., NG:NG + k] = True
        mv, _ = mesh.initialize(mv)
        out[name] = []
        for cycle in range(1, NSTEP + 1):
            mesh.set_cycle(cycle)
            if kk == 0:
                dts.append(mesh.max_time_step(mv))
            for stage in range(len(blocks[0].module("intg").stages)):
                mesh.forward(mv, dts[cycle - 1], stage)
            if mesh.check_redo(mv) != 0:
                raise RuntimeError(f"{name} step rejected (check_redo) at cycle {cycle}")
            out[name].append(torch.stack([v["hydro_w"][:5, NG:-NG, NG:-NG, NG + (k - kk):-NG].double()
                                          for v in mv], 1))
        del mesh, blocks, mv
    g = {}
    for s, w in zip(out["floor"], out["wall"]):
        g = worst(g, gap(s, w))
    return g


def terrain(scheme, device="cpu", nstep=10):
    """the six-panel deck at rest over terrain given on interior columns only (ghost depths are the
    API's): a 3-cell mountain centred on a panel edge and a 2-cell ridge along the equator across the
    equatorial panel edges; earth vs matched fill, the earth run's dt on both. Max |earth - matched|
    over the fluid per field and step (bitwise expected), and whether any state is non-finite."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_axial_momentum_flux_form as afm
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "test_axial_momentum_flux_form.yaml")))
    cfg["integration"]["implicit-scheme"] = scheme
    mesh, blocks = afm.build(cfg, device)
    c0 = blocks[0].module("coord")
    lo0, la0 = c0.buffer("longitude")[..., 0], c0.buffer("latitude")[..., 0]
    lon0 = float(lo0[NG + 4, -NG - 1] + lo0[NG + 4, -NG]) / 2  # on the panel's x2 edge
    lat0 = float(la0[NG + 4, -NG - 1] + la0[NG + 4, -NG]) / 2
    dlat = float((la0[NG + 1, NG + 4] - la0[NG, NG + 4]).abs())
    w0 = [v["hydro_w"].clone() for v in afm.initial_state(cfg, blocks, None, False)]
    masks = []
    for b in blocks:
        c = b.module("coord")
        lon, lat = c.buffer("longitude")[..., 0], c.buffer("latitude")[..., 0]
        dist = torch.arccos((torch.sin(lat) * math.sin(lat0) +
                             torch.cos(lat) * math.cos(lat0) * torch.cos(lon - lon0)).clamp(-1, 1))
        h = torch.round(3. * torch.exp(-(dist / (1.5 * dlat)) ** 2)).long()
        h = torch.maximum(h, torch.where(lat.abs() < .6 * dlat, 2, 0))
        s = torch.zeros(w0[0].shape[1:], dtype=torch.bool, device=w0[0].device)
        at = torch.arange(s.size(-1), device=s.device)
        s[NG:-NG, NG:-NG] = ((at >= NG) & (at < NG + h[NG:-NG, NG:-NG, None]))  # interior columns only
        masks.append(s)
    del mesh, blocks
    out, dts, bad = {}, [], False
    for fill in ("earth", "matched"):
        c = yaml.safe_load(yaml.safe_dump(cfg))
        sd, sp = (1.e3, 1.e9) if fill == "earth" else (float(w0[0][kIDN, NG, NG, NG]), float(w0[0][kIPR, NG, NG, NG]))
        c["boundary-condition"]["internal"] = {"solid-density": sd, "solid-pressure": sp}
        mesh, blocks = afm.build(c, device)
        mv, _ = mesh.initialize([{"hydro_w": w.clone(), "solid": s.clone()} for w, s in zip(w0, masks)])
        out[fill] = []
        for cycle in range(1, nstep + 1):
            mesh.set_cycle(cycle)
            if fill == "earth":
                dts.append(mesh.max_time_step(mv))
            for stage in range(len(blocks[0].module("intg").stages)):
                mesh.forward(mv, dts[cycle - 1], stage)
            fl = [(v["hydro_w"][:5], ~v["solid"]) for v in mv]  # ghost columns: the API's depth
            bad |= not all(bool(torch.isfinite(x[:, f]).all()) for x, f in fl)
            out[fill].append([x[:, f].double().cpu() for x, f in fl])
        del mesh, blocks, mv
    d = max(float((a - b).abs().max()) for sa, sb in zip(out["earth"], out["matched"]) for a, b in zip(sa, sb))
    return d, bad


def row_terrain():
    ok = True
    dev = [a for a in sys.argv[1:] if a in ("cpu", "cuda")] or ["cpu"]
    for device in dev:
        for scheme in (0, 9):
            tag = f"TERRAIN cubed-sphere mountain on a panel edge + equatorial ridge, scheme {scheme}, {device}"
            try:
                d, bad = terrain(scheme, device)
                ok &= d == 0. and not bad
                print(f"{tag}: max|earth - matched| {d:.2e} nonfinite {bad} " + ("PASS" if d == 0. and not bad else "FAIL"), flush=True)
            except RuntimeError as e:
                print(f"{tag}: RAISED {str(e).splitlines()[0][:300]} FAIL", flush=True)
                ok = False
    return ok


def row_sphere():
    tag = f"SPHERE cubed-sphere 2-cell floor vs wall, full, closure off, Coriolis off, {NSTEP} steps"
    try:
        return report(tag, sphere())
    except RuntimeError as e:
        print(f"{tag}: RAISED {str(e).splitlines()[0][:300]} FAIL", flush=True)
        return False


def main():
    rows = [a for a in sys.argv[1:] if a in ("cov", "conv4", "sphere", "terrain")] or ["cov", "conv4", "sphere", "terrain"]
    if len(rows) > 1:  # each row in its own process: two decks of different type in one process interfere
        rc = [subprocess.run([sys.executable, os.path.abspath(__file__), r]
                             + [a for a in sys.argv[1:] if a in ("cpu", "cuda")]).returncode for r in rows]
        ok = all(r == 0 for r in rc)
    elif rows[0] == "cov":
        ok = row_column("COV x1-mass-covariance", {"x1-mass-covariance": True})
    elif rows[0] == "terrain":
        ok = row_terrain()
    elif rows[0] == "conv4":
        ok = row_column("CONV4 conversion-order 4", {"conversion-order": 4})
    else:
        ok = row_sphere()
    print(f"GATE solid {' '.join(rows)}:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
