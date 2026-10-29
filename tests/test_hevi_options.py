#!/usr/bin/env python3
"""integration/implicit-integrator and its solve count, and parity of hevi-ark2 with the rk
path (implicit-integrator: rk) for the step-end operators.

Options:
  * implicit-scheme 9 under rk3 without the key runs hevi-ark2; implicit-integrator: rk opts out;
  * hevi-ark2 with implicit-scheme 1, 0 or absent and an unknown name are rejected;
  * the default is bitwise implicit-integrator-iterations 5 and differs from iterations 4;
    iterations 0 is rejected; the key is rejected under rk (named, or no implicit scheme 9)
    and accepted under implicit-scheme 9 without the integrator key;
    implicit-integrator-tolerance (the removed stop) is rejected as an unknown key.

Parity with the rk path:
  saturation  final-stage saturation adjustment (thermo with reactions). A supersaturated
              moist column at rest runs one step of SAT_DT on each path; both condense and
              max |cloud_hevi - cloud_rk| <= SAT_TOL max cloud_rk.
  solid       solid-cell refill. A moving isentropic column with 4 solid cells at the top runs
              NSTEP steps on each path; the solid cells hold their fill state bitwise and the
              fluid mass is conserved to MASS_TOL.

  python test_hevi_options.py [--device cuda]
"""
import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

SAT_TOL, MASS_TOL, NSTEP, SAT_DT = 1.0e-6, 1.0e-12, 20, 1.0e-3
DEVICE = "cpu"


def config(scheme, integrator, iterations=None):
    return {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": 1.e4, "x2min": 0., "x2max": 4.e3, "x3min": 0., "x3max": 1.e3},
                     "cells": {"nx1": 10, "nx2": 4, "nx3": 1, "nghost": 3}},
        "dynamics": {"equation-of-state": {"type": "ideal-gas", "gammad": 1.4, "weight": 0.029},
                     "reconstruct": {"vertical": {"type": "weno5", "scale": True},
                                     "horizontal": {"type": "weno5", "scale": True}},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.3, "nlim": -1, "tlim": 1.e9,
                        **({} if scheme is None else {"implicit-scheme": scheme}),
                        **({} if integrator is None else {"implicit-integrator": integrator}),
                        **({} if iterations is None else {"implicit-integrator-iterations": iterations})},
        "forcing": {"const-gravity": {"grav1": -9.8}},
    }


def make_block(cfg):
    from snapy import MeshBlock, MeshBlockOptions
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "deck.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(cfg, f)
        b = MeshBlock(MeshBlockOptions.from_yaml(p))
    b.to(torch.device(DEVICE), torch.float64)
    return b


def integrator_of(b):
    return b.options.hydro().icorr().integrator()


def options(fails):
    from snapy import kIDN, kIPR
    # one step of a column out of balance (uniform rho, p under gravity), so every stage
    # solve changes the result
    def step(iterations):
        blk = make_block(config(9, None, iterations))
        w0 = dict(blk.named_buffers())["hydro.D"].clone().zero_()
        w0[kIDN], w0[kIPR] = 1.0, 1.0e5
        v, _ = blk.initialize({"hydro_w": w0})
        for st in range(len(blk.module("intg").stages)):
            blk.forward(v, 5.0, st)
        return v["hydro_u"].clone(), integrator_of(blk)
    (u_default, name), (u5, _), (u4, _) = step(None), step(5), step(4)
    print("default integrator %s; default bitwise = iterations 5: %s; 5 differs from 4: %s "
          "(max |u5 - u4| %.3e)" % (name, torch.equal(u_default, u5), not torch.equal(u5, u4),
                                    (u5 - u4).abs().max().item()))
    if name != "hevi-ark2":
        fails.append("implicit-scheme 9 under rk3 without the key is %s" % name)
    if not torch.equal(u_default, u5):
        fails.append("hevi-ark2 default is not bitwise iterations 5")
    if torch.equal(u5, u4):
        fails.append("iterations 4 is bitwise iterations 5 (the key does not reach the solve)")
    if not torch.isfinite(u_default).all():
        fails.append("default step not finite")
    if integrator_of(make_block(config(9, "rk"))) != "rk":
        fails.append("implicit-integrator: rk does not select the rk path")

    def rejected(cfg, word):
        try:
            make_block(cfg)
        except Exception as e:  # TORCH_CHECK -> RuntimeError
            msg = next((l for l in str(e).splitlines() if word in l), "")
            print("rejected (%s): %s" % (word, msg[:150]))
            return bool(msg)
        return False
    if not rejected(config(9, "hevi-ark2", 0), "implicit-integrator-iterations"):
        fails.append("implicit-integrator-iterations 0 not rejected with its name")
    for scheme, name in ((1, "hevi-ark2"), (0, "hevi-ark2"), (None, "hevi-ark2"), (9, "ark-unknown")):
        if not rejected(config(scheme, name), name):
            fails.append("scheme %s with %s not rejected with its name" % (scheme, name))
    # the count only reaches hevi-ark2: rejected under rk (named, or no implicit scheme 9)
    for scheme, name in ((9, "rk"), (0, None), (None, "rk")):
        if not rejected(config(scheme, name, 3), "implicit-integrator-iterations"):
            fails.append("iterations under scheme %s, %s not rejected" % (scheme, name))
    make_block(config(9, None, 3))
    # the tolerance stop is gone: its key is an unknown key
    cfg = config(9, None)
    cfg["integration"]["implicit-integrator-tolerance"] = 1.e-10
    if not rejected(cfg, "implicit-integrator-tolerance"):
        fails.append("implicit-integrator-tolerance not rejected as an unknown key")


def saturation(integrator):
    from snapy import kICY, kIDN, kIPR
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent / "test_tracer_dry_convention.yaml"))
    cfg.pop("scalar")
    cfg["integration"]["implicit-scheme"] = 9
    if integrator:
        cfg["integration"]["implicit-integrator"] = integrator
    b = make_block(cfg)
    ng, nx1 = 3, cfg["geometry"]["cells"]["nx1"]
    dz = float(cfg["geometry"]["bounds"]["x1max"]) / nx1
    w = dict(b.named_buffers())["hydro.D"].clone().zero_()
    z = (torch.arange(w.size(-1), dtype=torch.float64) - ng + 0.5) * dz
    T, R = 270.0, 3900.0
    p = 1.0e5 * torch.exp(-z * 10.0 / (R * T))
    w[kIPR] = p.view(1, 1, -1)
    w[kIDN] = (p / (R * T)).view(1, 1, -1)
    w[kICY] = 0.3  # supersaturated vapour (about 5 kPa against about 0.5 kPa at 270 K)
    # the moist mixture's gas constant differs from R: rescale rho through the EOS to T
    Tw = b.module("hydro.eos").compute("W->T", [w])
    w[kIDN] = w[kIDN] * Tw / T
    v, _ = b.initialize({"hydro_w": w})
    for s in range(len(b.module("intg").stages)):
        b.forward(v, SAT_DT, s)
    I = (slice(None), slice(ng, -ng), slice(ng, -ng))
    return v["hydro_u"][kICY + 1][I].clone(), integrator_of(b)


def solid(integrator):
    import snapy
    from snapy import kIDN, kIPR, kIV1
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_gravity_work_fixer as box

    cfg = box.config({"gravity-work": "cell", "gravity-work-fixer": True}, scheme=9)
    if integrator:
        cfg["integration"]["implicit-integrator"] = integrator
    ng, nz = 3, box.NZ
    dz = box.LZ / nz
    z = (torch.arange(nz) + 0.5) * dz
    T = box.TS - box.GRAV * z / box.CP
    p = box.PS * (T / box.TS) ** (box.CP / box.RD)
    col = torch.zeros(kIPR + 1, 1, 1, nz, dtype=torch.float64)
    col[kIDN, 0, 0], col[kIPR, 0, 0] = p / (box.RD * T), p
    wb, _, _ = snapy.balance_column(col, torch.full((nz,), dz, dtype=torch.float64), box.GRAV)
    start = nz - 4
    cfg["boundary-condition"]["internal"] = {"solid-density": float(wb[kIDN, 0, 0, start]),
                                             "solid-pressure": float(wb[kIPR, 0, 0, start])}
    b = make_block(cfg)
    w = dict(b.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + nz] = wb[c, 0, 0]
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + nz:] = w[c][..., ng + nz - 1:ng + nz]
    w[kIV1][..., ng:ng + nz] = box.MACH * math.sqrt(box.GAMMA * box.RD * box.TS) * torch.sin(
        math.pi * z / box.LZ)
    sol = torch.zeros_like(w[kIDN], dtype=torch.bool)
    sol[..., ng + start:ng + start + 4] = True
    w[kIV1][sol] = 0
    for c, key in ((kIDN, "solid-density"), (kIPR, "solid-pressure")):
        w[c][sol] = cfg["boundary-condition"]["internal"][key]
    v, _ = b.initialize({"hydro_w": w, "solid": sol})
    m0 = v["hydro_u"][kIDN, ..., ng:ng + start].sum().item()
    dt = b.max_time_step(v)
    for _ in range(NSTEP):
        for s in range(len(b.module("intg").stages)):
            b.forward(v, dt, s)
    u = v["hydro_u"]
    fill = v["fill_solid_hydro_u"]
    exact = bool(torch.equal(u[:, sol], fill[:, sol]))
    mass = abs(u[kIDN, ..., ng:ng + start].sum().item() / m0 - 1)
    return exact, mass, bool(torch.isfinite(u).all()), integrator_of(b)


def parity(fails):
    ch, nh = saturation(None)
    cr, nr = saturation("rk")
    diff = (ch - cr).abs().max().item()
    scale = cr.abs().max().item()
    print("saturation: max cloud %s %.6e, %s %.6e; max |diff| %.3e (%.3e of max)"
          % (nh, ch.max().item(), nr, cr.max().item(), diff, diff / max(scale, 1e-300)))
    if (nh, nr) != ("hevi-ark2", "rk"):
        fails.append("saturation: integrator selection wrong (%s, %s)" % (nh, nr))
    if not (ch.max().item() > 0 and cr.max().item() > 0):
        fails.append("saturation: no condensate on one path")
    elif not diff <= SAT_TOL * scale:
        fails.append("saturation: paths differ by %.3e of max cloud" % (diff / scale))
    for want, integ in (("hevi-ark2", None), ("rk", "rk")):
        exact, mass, finite, name = solid(integ)
        print("solid %-9s: solid cells bitwise their fill state %s, fluid mass drift %.3e, "
              "finite %s" % (name, exact, mass, finite))
        if name != want:
            fails.append("solid: integrator %s, expected %s" % (name, want))
        if not (exact and finite and mass <= MASS_TOL):
            fails.append("solid %s: exact %s finite %s mass %.3e" % (name, exact, finite, mass))


def main(argv=None):
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 125
    DEVICE = args.device
    torch.set_default_dtype(torch.float64)
    fails = []
    options(fails)
    parity(fails)
    print("FAILED: " + "; ".join(fails) if fails else "PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
