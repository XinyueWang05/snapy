"""LMARS energy-flux stability test: one arm.

2-D Cartesian, x1 vertical (450 km, 100 levels, reflecting), x2 horizontal
periodic (NX2 columns of width dx), dry H2-He, g = 8.87, isothermal T0 = 100 K,
column projected onto the solver's discrete hydrostatic balance with
snapy.balance_column, then 1e-6 K Gaussian white noise in T (fixed seed, p held
fixed) and a uniform wind u in x2. The deck is derived from
examples/jupiter_crm_dry.yaml (species, EOS, reconstruction).

Writes <out>/series.csv (t, rms w, max |w|, rms u', ...) every --dt-diag
seconds, w snapshots at t = 0, 1, 2, 3 d (and at the stop time), and
<out>/summary.json.

usage: python run_arm.py --solver lmars --u 400 --dx 5e3 --out runs/lmars_u400_dx5
       [--days 3] [--noise 1e-6] [--seed 20261025] [--nx2 64] [--dt-diag 3600]
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import yaml

import snapy
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR

p = argparse.ArgumentParser()
p.add_argument("--solver", required=True, choices=["lmars", "hllc"])
p.add_argument("--u", type=float, required=True, help="uniform x2 wind [m/s]")
p.add_argument("--dx", type=float, required=True, help="x2 cell width [m]")
p.add_argument("--out", required=True)
p.add_argument("--days", type=float, default=3.0)
p.add_argument("--noise", type=float, default=1.0e-6, help="std of T noise [K]")
p.add_argument("--seed", type=int, default=20261025)
p.add_argument("--nx2", type=int, default=64)
p.add_argument("--nx1", type=int, default=100)
p.add_argument("--ztop", type=float, default=450.0e3)
p.add_argument("--T0", type=float, default=100.0)
p.add_argument("--ps", type=float, default=1.0e6, help="bottom-cell pressure guess [Pa]")
p.add_argument("--grav", type=float, default=8.87)
p.add_argument("--cfl", type=float, default=0.3)
p.add_argument("--scheme", type=int, default=9, help="integration/implicit-scheme")
p.add_argument("--dt-diag", type=float, default=3600.0)
p.add_argument("--wall-limit", type=float, default=0.0,
               help="stop after this many wall seconds (0 = none); recorded in summary")
a = p.parse_args()

torch.set_num_threads(1)
os.makedirs(a.out, exist_ok=True)
DAY = 86400.0
NG = 3
cfg = {
    "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
    "species": [{"name": "dry", "composition": {"H": 1.5, "He": 0.15}, "cv_R": 2.5}],
    "geometry": {
        "type": "cartesian",
        "bounds": {"x1min": 0.0, "x1max": a.ztop, "x2min": 0.0, "x2max": a.nx2 * a.dx,
                   "x3min": -0.5, "x3max": 0.5},
        "cells": {"nx1": a.nx1, "nx2": a.nx2, "nx3": 1, "nghost": NG},
    },
    "dynamics": {
        "equation-of-state": {"type": "ideal-moist", "density-floor": 1.0e-10,
                              "pressure-floor": 1.0e-10, "limiter": True,
                              "max-iter": 10, "ftol": 1.0e-6},
        "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": True},
                        "horizontal": {"type": "weno5", "scale": True, "shock": True}},
        "riemann-solver": {"type": a.solver},
    },
    "boundary-condition": {"external": {
        "x1-inner": "reflecting", "x1-outer": "reflecting",
        "x2-inner": "periodic", "x2-outer": "periodic",
        "x3-inner": "periodic", "x3-outer": "periodic"}},
    "integration": {"type": "rk3", "cfl": a.cfl, "implicit-scheme": a.scheme,
                    "nlim": -1, "tlim": a.days * DAY},
    "forcing": {"const-gravity": {"grav1": -a.grav}},
}
yml = os.path.join(a.out, "deck.yaml")
yaml.safe_dump(cfg, open(yml, "w"), sort_keys=False)

op = MeshBlockOptions.from_yaml(yml)
op.output_dir(a.out)
block = MeshBlock(op)
block.to(torch.device("cpu"), torch.float64)
eos = block.module("hydro.eos")

x1v = block.buffer("coord.x1v")
dx1f = block.buffer("coord.dx1f")
il, iu = NG, NG + a.nx1  # interior x1 slice [il, iu)
w = block.buffer("hydro.D").clone().zero_()  # (nvar, nc3, nc2, nc1)
nvar = w.shape[0]

# gas constant from the EOS itself: W->T is p/(rho R) for this dry gas
probe = w.clone()
probe[kIDN] = 1.0
probe[kIPR] = 1.0e5
Rd = 1.0e5 / eos.compute("W->T", [probe]).flatten()[0].item()

# continuum isothermal column, then the discrete balance at fixed T
z = x1v[il:iu]
pcol = a.ps * torch.exp(-a.grav * (z - z[0]) / (Rd * a.T0))
wi = torch.zeros(nvar, 1, w.shape[2], a.nx1, dtype=torch.float64)
wi[kIPR] = pcol
wi[kIDN] = pcol / (Rd * a.T0)
wb, resid, sweeps = snapy.balance_column(wi.contiguous(), dx1f[il:iu].contiguous(),
                                         a.grav, True, 1.0e-12, 400, "cartesian")
w[..., il:iu] = wb
# ghosts: fill with the edge values; initialize() applies the boundary conditions
w[..., :il] = wb[..., :1]
w[..., iu:] = wb[..., -1:]

# white noise in T, p held fixed: rho -> rho T0 / (T0 + T')
gen = torch.Generator().manual_seed(a.seed)
dT = a.noise * torch.randn((1, a.nx2, a.nx1), generator=gen, dtype=torch.float64)
w[kIDN, :, NG:NG + a.nx2, il:iu] *= a.T0 / (a.T0 + dT)
w[2] = a.u  # x2 velocity (IVY)

bv, t = block.initialize({"hydro_w": w})


def interior(f):
    return f[0, NG:NG + a.nx2, il:iu]  # (nx2, nx1)


def diag(t):
    ww = bv["hydro_w"]
    wv = interior(ww[1])
    uv = interior(ww[2]) - a.u
    T = interior(eos.compute("W->T", [ww]))
    return {"t": t, "rms_w": wv.square().mean().sqrt().item(), "max_w": wv.abs().max().item(),
            "rms_du": uv.square().mean().sqrt().item(),
            "rms_dT": (T - T.mean(0, keepdim=True)).square().mean().sqrt().item()}


snaps = {}


def snap(t, tag):
    snaps[tag] = {"t": t, "w": interior(bv["hydro_w"][1]).clone().numpy(),
                  "T": interior(eos.compute("W->T", [bv["hydro_w"]])).clone().numpy()}
    np.savez(os.path.join(a.out, "snaps.npz"),
             **{f"{k}_{v}": snaps[k][v] for k in snaps for v in ("w", "T")},
             **{f"{k}_t": snaps[k]["t"] for k in snaps},
             x1=x1v[il:iu].numpy())


series = [diag(t)]
snap(t, "d0")
fcsv = open(os.path.join(a.out, "series.csv"), "w")
fcsv.write("t,rms_w,max_w,rms_du,rms_dT,cycle,wall\n")


def wrow(d, cyc, wall):
    fcsv.write("%.10g,%.6e,%.6e,%.6e,%.6e,%d,%.1f\n" % (d["t"], d["rms_w"], d["max_w"],
                                                      d["rms_du"], d["rms_dT"], cyc, wall))
    fcsv.flush()


wrow(series[0], 0, 0.0)
tend = a.days * DAY
next_diag = a.dt_diag
next_day = DAY
status, nredo, dts = "completed", 0, []
w0 = time.time()
while t < tend * (1 - 1e-14):
    dt = block.max_time_step(bv)
    dt = min(dt, next_diag - t)
    for st in range(len(block.intg.stages)):
        block.forward(bv, dt, st)
    err = block.check_redo(bv)
    if err > 0:
        nredo += 1
        if nredo > 1000:
            status = "too many redos"
            break
        continue
    if err < 0:
        status = "check_redo terminated"
        break
    block.inc_cycle()
    t += dt
    dts.append(dt)
    if not torch.isfinite(bv["hydro_w"]).all():
        status = "non-finite state"
        break
    if t >= next_diag * (1 - 1e-12):
        d = diag(t)
        series.append(d)
        wrow(d, block.cycle(), time.time() - w0)
        next_diag += a.dt_diag
        if t >= next_day * (1 - 1e-12):
            snap(t, "d%d" % round(t / DAY))
            next_day += DAY
        if a.wall_limit and time.time() - w0 > a.wall_limit:
            status = "wall limit"
            break
wall = time.time() - w0
if status != "completed" or "d%d" % round(t / DAY) not in snaps:
    if torch.isfinite(bv["hydro_w"]).all():
        snap(t, "stop")
summ = {"solver": a.solver, "u": a.u, "dx": a.dx, "nx2": a.nx2, "nx1": a.nx1,
        "status": status, "t_end": t, "days_req": a.days, "cycles": block.cycle(),
        "redos": nredo, "wall_s": wall, "wall_s_per_model_hour": wall / max(t / 3600.0, 1e-30),
        "dt_min": min(dts) if dts else None, "dt_max": max(dts) if dts else None,
        "Rd": Rd, "balance_residual": resid, "balance_sweeps": sweeps,
        "noise": a.noise, "seed": a.seed, "T0": a.T0, "grav": a.grav, "ztop": a.ztop, "scheme": a.scheme, "cfl": a.cfl,
        "snapy_file": snapy.__file__, "torch": torch.__version__}
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print(json.dumps(summ))
