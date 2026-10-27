#!/usr/bin/env python3
"""integration/implicit-integrator: rk (default) or hevi-ark2, the latter only with vic-full.

  * hevi-ark2 with implicit-scheme 9 builds a block and runs one step on a rest column;
  * hevi-ark2 with implicit-scheme 1, 0 or absent and an unknown name are rejected at
    construction;
  * without implicit-integrator-iterations, hevi-ark2 runs 3 stage iterations: a step is
    bitwise the step with the key set to 3 and differs from the key set to 1;
    implicit-integrator-iterations 0 is rejected;
  * implicit-integrator-iterations under rk (named or default, implicit-scheme 9, 0 or
    absent) is rejected at construction.

  python test_hevi_ark2_options.py
"""
import os
import sys
import tempfile

import torch
import yaml


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


def build(scheme, integrator, tmp, iterations=None):
    from snapy import MeshBlock, MeshBlockOptions
    p = os.path.join(tmp, "deck.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(config(scheme, integrator, iterations), f)
    return MeshBlock(MeshBlockOptions.from_yaml(p))


def main():
    torch.set_default_dtype(torch.float64)
    from snapy import kIDN, kIPR
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        b = build(9, "hevi-ark2", tmp)
        b.to(torch.device("cpu"))
        w = dict(b.named_buffers())["hydro.D"].clone().zero_()
        w[kIDN], w[kIPR] = 1.0, 1.0e5
        bv, _ = b.initialize({"hydro_w": w})
        for s in range(len(b.module("intg").stages)):
            b.forward(bv, 0.1, s)
        if not torch.isfinite(bv["hydro_u"]).all():
            fails.append("hevi-ark2 step not finite")
        # default iterations: one step of a column out of balance (uniform rho, p
        # under gravity), so every stage iteration changes the result
        def step(iterations):
            blk = build(9, "hevi-ark2", tmp, iterations)
            blk.to(torch.device("cpu"))
            w0 = dict(blk.named_buffers())["hydro.D"].clone().zero_()
            w0[kIDN], w0[kIPR] = 1.0, 1.0e5
            v, _ = blk.initialize({"hydro_w": w0})
            for st in range(len(blk.module("intg").stages)):
                blk.forward(v, 5.0, st)
            return v["hydro_u"].clone()
        u_default, u3, u1 = step(None), step(3), step(1)
        same3, diff1 = torch.equal(u_default, u3), not torch.equal(u_default, u1)
        print("default iterations: bitwise = 3: %s; differs from 1: %s (max |u3 - u1| %.3e)"
              % (same3, diff1, (u3 - u1).abs().max().item()))
        if not (same3 and diff1):
            fails.append("hevi-ark2 default is not 3 stage iterations")
        try:
            build(9, "hevi-ark2", tmp, 0)
            fails.append("accepted implicit-integrator-iterations 0")
        except Exception as e:
            print("rejected iterations 0: %s" % str(e).splitlines()[0][:100])
        for scheme, name in ((1, "hevi-ark2"), (0, "hevi-ark2"), (None, "hevi-ark2"),
                             (9, "ark-unknown")):
            try:
                build(scheme, name, tmp)
                fails.append("accepted scheme %s with %s" % (scheme, name))
            except Exception as e:  # TORCH_CHECK -> RuntimeError
                msg = next((l for l in str(e).splitlines() if name in l), "")
                print("rejected scheme %s, %s: %s" % (scheme, name, msg[:160]))
                if not msg:
                    fails.append("scheme %s with %s rejected without naming %s" % (scheme, name, name))
        # implicit-integrator-iterations only reaches hevi-ark2: under rk (named or
        # by default, with or without an implicit scheme) it is rejected
        key = "implicit-integrator-iterations"
        for scheme, name in ((9, "rk"), (9, None), (0, None), (None, "rk")):
            try:
                build(scheme, name, tmp, 3)
                fails.append("accepted %s 3 with scheme %s, %s" % (key, scheme, name))
            except Exception as e:
                msg = next((l for l in str(e).splitlines() if key in l), "")
                print("rejected %s 3 with scheme %s, %s: %s" % (key, scheme, name, msg[:160]))
                if not msg:
                    fails.append("%s 3 with scheme %s, %s rejected without naming %s"
                                 % (key, scheme, name, key))
    if fails:
        print("FAILED:", "; ".join(fails))
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
