#!/usr/bin/env python3
"""Time-stepped Cartesian and spherical x1 face-work columns (#288).

The discretely balanced 11.3H column must remain at rest through the tested
vertical acoustic Courants up to 250. Results include the first failed rung;
passing a finite ladder does not establish a universal stability threshold.
Cartesian retains eight periodic copies; spherical x1 uses one angular cell
to test a strictly radial column. Angular-mode stability is a separate gate.
--geometry gnomonic-equiangle runs the same ladder on one cubed-sphere face block of
4 x 4 columns at the spherical radius, where the corrected-PE work carries the
centroid term of the derivation's sec 12 (#300).

The face ladder runs twice, each in a child process (the switch is read once per
process): with SNAP_GRAVITY_WORK_RADIAL_EXACT=0, and with it on (the corrected-PE
work of docs/derivations/curved-gravity-work-weight.md sec 7, booked inside the
implicit operator, #296). Every rung must stay below W_TOL, end the run below
W_SETTLED, and, with the switch on, peak at most ON_OFF times its switch-off rung (or
W_ROUNDOFF, the round-off level of the rebalanced column).
A child that exits non-zero or prints an incomplete ladder fails the parent (#299).

Every arm starts from the column rebalanced against the run's own discrete operator
(rebalance). snapy.balance_column (rtol 1e-12, 400 sweeps) leaves a rest residual of
~5.5e-13 (scaled by rho g) against that operator; at its default rtol 1e-10 it left 3.3e-11
in the bottom levels, which seeded an acoustic transient that explicit scheme 0, converged
in dt, carried to max w ~2.0e-6. One Gauss-Newton step on (ln rho, ln p) of the column,
with the explicit tendency of one tiny step from rest as the residual, takes it to
round-off (~8e-15) and the explicit transient to ~3e-10.

With SNAP_TEST_IMPLICIT_INTEGRATOR=hevi-ark2 (the _hevi entry) the face ladder runs the
hevi-ark2 rungs HEVI_COURANTS in one child at the default gravity work, i.e. with
SNAP_GRAVITY_WORK_RADIAL_EXACT unset (on), each without an early stop over at least
HEVI_T_END (the 40 steps of the 6.6 rung), with the default stage solves (4). Every rung
must stay finite with max w at most HEVI_W_TOL on every step: from the rebalanced column
the run must stay at round-off (4 solves peak at 2.1e-10; 3 solves grow to 2.9e-9 at
Courant 197/250 by step 40). That bar replaces W_TOL,
W_SETTLED and ON_OFF in this arm. The cell-work + global-fixer arm runs on the rk ladder
only: hevi-ark2 rejects the gravity-work fixer at its first step. The rk ladder keeps
W_TOL, W_SETTLED and ON_OFF.
"""
import argparse
import math
import json
import os
import subprocess
import sys
import tempfile

import torch
import yaml

GRAV, GAMMA, RD, T0, PS = 23.1, 1.403461, 3515.0, 786.0, 1.0e7
NZ, DZ, NX2 = 45, 29946.8085106, 8
COURANTS = (6.6, 65.6, 100.0, 197.0, 250.0)
RADIUS = 7.e7
W_TOL = 1.0e-7  # m/s
W_SETTLED = 1.0e-10  # m/s, max w at the last step: the column has settled
ON_OFF = 1.1  # switch on: max w at most this times the switch-off rung's,
W_ROUNDOFF = 1.0e-11  # m/s round-off floor: measured 8.1e-13 (OFF) / 8.2e-13 (ON) at C 100
# hevi-ark2 arm (see the docstring)
HEVI = os.environ.get("SNAP_TEST_IMPLICIT_INTEGRATOR", "rk") != "rk"
HEVI_COURANTS = (0.4, 1.65, 3.3) + COURANTS
HEVI_W_TOL = 1.0e-9  # m/s, max w on every step of every hevi rung (4.7x the 2.1e-10 peak)

def config(geometry="cartesian", cell_work=False):
    rgas = 8.31446261815324
    cfg = {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": NZ * DZ, "x2min": 0.0, "x2max": 31415926.53589793,
                                "x3min": 0.0, "x3max": 3926990.8169872416},
                     "cells": {"nx1": NZ, "nx2": NX2, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": rgas / RD,
                                  "density-floor": 3.4682e-18, "pressure-floor": 3.4682e-18,
                                  "temperature-floor": 1.e-6, "limiter": True},
            "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                            "horizontal": {"type": "weno5", "scale": True, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.5, "implicit-scheme": 9, "nlim": -1, "tlim": 1.e9,
                        # the _hevi ctest entries run the ladder with hevi-ark2
                        "implicit-integrator": os.environ.get("SNAP_TEST_IMPLICIT_INTEGRATOR", "rk")},
        "forcing": {"const-gravity": {"grav1": -GRAV, "gravity-work": "face",
                                          "gravity-work-fixer": False}},
    }

    if cell_work:
        # Cell work, with the fixer key omitted: the global fixer is on by default with cell.
        cfg["forcing"]["const-gravity"]["gravity-work"] = "cell"
        cfg["forcing"]["const-gravity"].pop("gravity-work-fixer")

    if geometry == "spherical-polar":
        cfg["geometry"]["type"] = geometry
        cfg["geometry"]["cells"]["nx2"] = 1
        cfg["geometry"]["bounds"] = {
            "x1min": RADIUS, "x1max": RADIUS + NZ * DZ,
            "x2min": math.pi / 2 - 0.2, "x2max": math.pi / 2 + 0.2,
            "x3min": 0.0, "x3max": 0.05}
    elif geometry == "gnomonic-equiangle":
        cfg["geometry"]["type"] = geometry
        cfg["geometry"]["cells"].update(nx2=4, nx3=4)
        cfg["geometry"]["bounds"] = {
            "x1min": RADIUS, "x1max": RADIUS + NZ * DZ,
            "x2min_pi": -0.25, "x2max_pi": 0.25, "x3min_pi": -0.25, "x3max_pi": 0.25}
        for side in ("x2-inner", "x2-outer", "x3-inner", "x3-outer"):
            cfg["boundary-condition"]["external"][side] = "reflecting"
    return cfg


def interior_of(geometry):
    ng, cells = 3, config(geometry)["geometry"]["cells"]
    jl = ng if cells["nx2"] > 1 else 0
    kl = ng if cells["nx3"] > 1 else 0
    return (slice(None), slice(kl, kl + cells["nx3"]), slice(jl, jl + cells["nx2"]), slice(ng, ng + NZ))


def initialize(block, rho, p):
    """the block at rest on the column (rho, p), ghosts copied from the edge cells"""
    from snapy import kIDN, kIPR
    ng = 3
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    for c, v in ((kIDN, rho), (kIPR, p)):
        w[c][..., ng:ng + NZ] = v.to(w)
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ:] = w[c][..., ng + NZ - 1:ng + NZ]
    block_vars, _ = block.initialize({"hydro_w": w})
    return block_vars


def explicit_block(geometry, device, cell_work=False):
    from snapy import MeshBlock, MeshBlockOptions
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        cfg = config(geometry, cell_work)
        cfg["integration"].update({"implicit-scheme": 0, "implicit-integrator": "rk"})
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    return block


_REBALANCED = {}


def rebalance(geometry, device, cell_work, rho, p, h=1.0, eps=1.0e-7, newton=3, tol=1.0e-13):
    """the column zeroing the explicit operator's own rest tendency: Gauss-Newton on
    (ln rho, ln p), residual = the change of (rho, m1, E) over one explicit step of h
    seconds from rest, scaled by rho cs / dz, rho g and p cs / dz, over every column"""
    from snapy import kIDN, kIPR, kIV1
    key = (geometry, device, cell_work, os.environ.get("SNAP_GRAVITY_WORK_RADIAL_EXACT"))
    if key in _REBALANCED:
        return _REBALANCED[key]
    block = explicit_block(geometry, device, cell_work)
    interior = interior_of(geometry)
    nstage = len(block.module("intg").stages)
    cs = math.sqrt(GAMMA * RD * T0)

    def residual(x):
        r_, p_ = x[:NZ].exp(), x[NZ:].exp()
        bv = initialize(block, r_, p_)
        u0 = bv["hydro_u"][interior].clone()
        for stage in range(nstage):
            block.forward(bv, h, stage)
        du = ((bv["hydro_u"][interior] - u0) / h).cpu()
        return torch.cat([(du[kIDN] / (r_ * cs / DZ)).flatten(), (du[kIV1] / (r_ * GRAV)).flatten(),
                          (du[kIPR] / (p_ * cs / DZ)).flatten()])

    x = torch.cat([rho.log(), p.log()])
    for _ in range(newton):
        r = residual(x)
        if r.abs().max().item() < tol:
            break
        jac = torch.empty(r.numel(), 2 * NZ, dtype=torch.float64)
        for j in range(2 * NZ):
            xj = x.clone()
            xj[j] += eps
            jac[:, j] = (residual(xj) - r) / eps
        x = x + torch.linalg.lstsq(jac, -r.unsqueeze(1), driver="gelsd").solution.squeeze(1)
    _REBALANCED[key] = (x[:NZ].exp(), x[NZ:].exp())
    print(json.dumps({"geometry": geometry, "cell_work": cell_work, "rebalance_residual_before": residual(
        torch.cat([rho.log(), p.log()])).abs().max().item(),
        "rebalance_residual_after": residual(x).abs().max().item()}), flush=True)
    return _REBALANCED[key]


def run(dt, nstep, device, geometry="cartesian", cell_work=False, stop=True, integration=None):
    """stop=False: no early stop at W_TOL, and history holds max w at every step;
    integration overrides the deck's integration keys."""

    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        cfg = config(geometry, cell_work)
        cfg["integration"].update(integration or {})
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))

    z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * DZ
    p = PS * torch.exp(-GRAV * z / (RD * T0))
    col = torch.zeros(kIPR + 1, 1, 1, NZ, dtype=torch.float64)
    col[kIDN, 0, 0] = p / (RD * T0)
    col[kIPR, 0, 0] = p
    wb, err, _ = snapy.balance_column(col, torch.full((NZ,), DZ, dtype=torch.float64), GRAV,
                                      rtol=1.e-12, max_iter=400)
    rho, p = rebalance(geometry, device, cell_work, wb[kIDN, 0, 0].clone(), wb[kIPR, 0, 0].clone())
    block_vars = initialize(block, rho, p)
    interior = interior_of(geometry)
    nstage = len(block.module("intg").stages)
    wmax, history = 0.0, []
    for n in range(1, nstep + 1):
        for stage in range(nstage):
            block.forward(block_vars, dt, stage)
        redo = block.check_redo(block_vars)
        if redo:
            return False, n, wmax, history, err
        u = block_vars["hydro_u"][interior]
        if not torch.isfinite(u).all():
            return False, n, wmax, history, err
        wn = (u[kIV1] / u[kIDN]).abs().max().item()
        wmax = max(wmax, wn)
        if not stop:
            history.append(wn)
            continue
        if wmax >= W_TOL:
            return False, n, wmax, history, err
        if n in (1, 10, 20, 30, nstep):
            history.append((n, wn))
    return True, nstep, wmax, history, err


def hevi_t_end(cs):
    """the simulated time of the 40-step 6.6 rung, 4015 s"""
    return 40 * COURANTS[0] * DZ / cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nstep", type=int, default=40)
    ap.add_argument("--geometry", choices=("cartesian", "spherical-polar", "gnomonic-equiangle", "both"),
                    default="both")
    ap.add_argument("--courants", type=float, nargs="+", default=HEVI_COURANTS if HEVI else COURANTS)
    ap.add_argument("--ladder", action="store_true",
                    help="child: the face ladder only, under the inherited switch")
    args = ap.parse_args()
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 0
    torch.set_default_dtype(torch.float64)

    cs = math.sqrt(GAMMA * RD * T0)
    failures = []
    geometries = ("cartesian", "spherical-polar") if args.geometry == "both" else (args.geometry,)
    if not args.ladder:
        ladder = {}
        # rk: SNAP_GRAVITY_WORK_RADIAL_EXACT off, on; hevi-ark2: unset, the default (on)
        for value in ((None,) if HEVI else ("0", "1")):
            env = dict(os.environ)
            env.pop("SNAP_GRAVITY_WORK_RADIAL_EXACT", None)
            if value is not None:
                env["SNAP_GRAVITY_WORK_RADIAL_EXACT"] = value
            cmd = [sys.executable, os.path.abspath(__file__), "--ladder", "--device", args.device,
                   "--nstep", str(args.nstep), "--geometry", args.geometry,
                   "--courants"] + [str(c) for c in args.courants]
            out = subprocess.run(cmd, env=env, capture_output=True, text=True)
            rungs, summaries = set(), set()
            for line in out.stdout.splitlines():
                if not line.startswith("{"):
                    continue
                row = json.loads(line)
                row["radial_exact"] = value != "0"
                print(json.dumps(row), flush=True)
                if "first_failing_courant" in row:
                    summaries.add(row["geometry"])
                if "passed" not in row:
                    continue
                rungs.add((row["geometry"], row["courant"]))
                arm = "radial-exact" if value != "0" else "face"
                ladder[arm, row["geometry"], row["courant"]] = row
                if not row["passed"]:
                    failures.append((arm, row["geometry"], row["courant"], row["steps"], row["max_w"]))
                elif not HEVI and row["history"][-1][1] > W_SETTLED:
                    failures.append((arm, row["geometry"], row["courant"], "not settled",
                                     row["history"][-1]))
            # a child exits 1 for a failing rung (already listed above) and for an
            # exception alike, so any non-zero exit fails, and the ladder must be complete
            missing = [(g, c) for g in geometries for c in args.courants if (g, c) not in rungs]
            missing += [(g, "summary") for g in geometries if g not in summaries]
            if out.returncode != 0 or missing:
                print(out.stderr[-2000:], flush=True)
                failures.append(("child", value, out.returncode, "missing rows", missing))
        for (arm, geometry, courant), on in ladder.items():
            off = ladder.get(("face", geometry, courant))
            if not HEVI and arm == "radial-exact" and off and on["max_w"] > max(ON_OFF * off["max_w"], W_ROUNDOFF):
                failures.append((arm, geometry, courant, "max w", on["max_w"], "off", off["max_w"]))
    for geometry in (geometries if args.ladder else ()):
        first_failure = None
        for courant in args.courants:
            dt = courant * DZ / cs
            if HEVI:
                nstep = max(args.nstep, math.ceil(hevi_t_end(cs) / dt))
                finite, n, wmax, w, err = run(dt, nstep, args.device, geometry, stop=False)
                passed = finite and n == nstep and wmax <= HEVI_W_TOL
                k = w.index(max(w)) if w else -1
                print(json.dumps({"geometry": geometry, "courant": courant, "dt": dt, "steps": n,
                                  "passed": passed, "max_w": wmax, "peak_step": k + 1,
                                  "last_w": w[-1] if w else None, "w_tol": HEVI_W_TOL,
                                  "balance_error": err, "history": w}), flush=True)
            else:
                finite, n, wmax, history, err = run(dt, args.nstep, args.device, geometry)
                passed = finite and wmax < W_TOL
                print(json.dumps({"geometry": geometry, "courant": courant, "dt": dt,
                                  "steps": n, "passed": passed, "max_w": wmax,
                                  "balance_error": err, "history": history}), flush=True)
            if not passed:
                if first_failure is None:
                    first_failure = courant
                failures.append((geometry, courant, n, wmax))
        print(json.dumps({"geometry": geometry, "first_failing_courant": first_failure,
                          "tested_courants": args.courants}), flush=True)
    # Preserve the Cartesian cell work + global fixer coverage alongside the
    # explicitly face-only, fixer-off ladder (rk only: hevi-ark2 rejects the fixer).
    if HEVI and not args.ladder:
        print("SKIP cell-global-fixer: hevi-ark2 does not support the gravity-work fixer", flush=True)
    for dt in (() if args.ladder or HEVI else (997.0, 100.0)):
        finite, n, wmax, history, err = run(
            dt, args.nstep, args.device, cell_work=True)
        passed = finite and wmax < W_TOL
        print(json.dumps({"arm": "cell-global-fixer", "dt": dt,
                          "steps": n, "passed": passed, "max_w": wmax,
                          "balance_error": err, "history": history}), flush=True)
        if not passed:
            failures.append(("cell-global-fixer", dt, n, wmax))
    for failure in failures:
        print("FAIL", failure, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
