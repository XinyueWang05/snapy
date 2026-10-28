#!/usr/bin/env python3
"""integration/implicit-integrator: rk or hevi-ark2, the latter only with vic-full; without
the key, implicit-scheme 9 under rk3 runs hevi-ark2 (implicit-integrator: rk opts out).

  * hevi-ark2 with implicit-scheme 9 builds a block and runs one step on a rest column;
  * hevi-ark2 with implicit-scheme 1, 0 or absent and an unknown name are rejected at
    construction;
  * the fixed-solve fallback: implicit-integrator-tolerance 0 with iterations 4 is bitwise
    iterations 4 alone and differs from iterations 3; the default (4 fixed solves) step is
    finite; implicit-integrator-iterations 0 and a negative tolerance are rejected;
  * implicit-integrator-iterations and -tolerance under rk (named, or no implicit scheme
    9) are rejected at construction; under implicit-scheme 9 without the integrator key
    (hevi-ark2 by default) they are accepted.

  python test_hevi_ark2_options.py
"""
import os
import sys
import tempfile

import torch
import yaml


def config(scheme, integrator, iterations=None, tolerance=None):
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
                        **({} if iterations is None else {"implicit-integrator-iterations": iterations}),
                        **({} if tolerance is None else {"implicit-integrator-tolerance": tolerance})},
        "forcing": {"const-gravity": {"grav1": -9.8}},
    }


def build(scheme, integrator, tmp, iterations=None, tolerance=None):
    from snapy import MeshBlock, MeshBlockOptions
    p = os.path.join(tmp, "deck.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(config(scheme, integrator, iterations, tolerance), f)
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
        def step(iterations, tolerance=None):
            blk = build(9, "hevi-ark2", tmp, iterations, tolerance)
            blk.to(torch.device("cpu"))
            w0 = dict(blk.named_buffers())["hydro.D"].clone().zero_()
            w0[kIDN], w0[kIPR] = 1.0, 1.0e5
            v, _ = blk.initialize({"hydro_w": w0})
            for st in range(len(blk.module("intg").stages)):
                blk.forward(v, 5.0, st)
            return v["hydro_u"].clone(), blk.hevi_solve_stats().tolist()
        (u_default, st_default), (u4, _), (u3, _), (u40, st40) = (
            step(None), step(4), step(3), step(4, 0.0))
        same4, diff3 = torch.equal(u40, u4), not torch.equal(u4, u3)
        print("fixed solves: tolerance 0 + iterations 4 bitwise = iterations 4: %s (stats %s); "
              "4 differs from 3: %s (max |u4 - u3| %.3e); default (4 fixed solves) stats %s, "
              "max |u_default - u4| %.3e" % (same4, st40, diff3, (u4 - u3).abs().max().item(),
                                             st_default, (u_default - u4).abs().max().item()))
        if not (same4 and diff3):
            fails.append("the fixed-solve fallback is not 4 stage solves")
        if not (torch.isfinite(u_default).all() and st_default[0] > 0):
            fails.append("default step not finite or not hevi-ark2")
        try:
            build(9, "hevi-ark2", tmp, 0)
            fails.append("accepted implicit-integrator-iterations 0")
        except Exception as e:
            print("rejected iterations 0: %s" % str(e).splitlines()[0][:100])
        try:
            build(9, "hevi-ark2", tmp, None, -1.0)
            fails.append("accepted implicit-integrator-tolerance -1")
        except Exception as e:
            print("rejected tolerance -1: %s" % str(e).splitlines()[0][:100])
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
        # the solve keys only reach hevi-ark2: under rk (named, or no implicit scheme 9)
        # they are rejected; under implicit-scheme 9 without the integrator key (hevi-ark2
        # by default) they are accepted
        for key, val in (("implicit-integrator-iterations", 3),
                         ("implicit-integrator-tolerance", 1.e-8)):
            kw = {"iterations": val} if "iterations" in key else {"tolerance": val}
            for scheme, name in ((9, "rk"), (0, None), (None, "rk")):
                try:
                    build(scheme, name, tmp, **kw)
                    fails.append("accepted %s with scheme %s, %s" % (key, scheme, name))
                except Exception as e:
                    msg = next((l for l in str(e).splitlines() if key in l), "")
                    print("rejected %s with scheme %s, %s: %s" % (key, scheme, name, msg[:160]))
                    if not msg:
                        fails.append("%s with scheme %s, %s rejected without naming %s"
                                     % (key, scheme, name, key))
            try:
                build(9, None, tmp, **kw)
                print("accepted %s with scheme 9 and no integrator key (hevi-ark2 default)" % key)
            except Exception as e:
                fails.append("rejected %s with scheme 9, no key: %s"
                             % (key, str(e).splitlines()[0][:100]))
    if fails:
        print("FAILED:", "; ".join(fails))
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
