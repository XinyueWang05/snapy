#!/usr/bin/env python3
"""integration/implicit-integrator: rk (default) or hevi-ark2, the latter only with vic-full.

  * hevi-ark2 with implicit-scheme 9 builds a block and runs one step on a rest column;
  * hevi-ark2 with implicit-scheme 1 and an unknown name are rejected at construction.

  python test_hevi_ark2_options.py
"""
import os
import sys
import tempfile

import torch
import yaml


def config(scheme, integrator):
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
        "integration": {"type": "rk3", "cfl": 0.3, "implicit-scheme": scheme, "nlim": -1, "tlim": 1.e9,
                        "implicit-integrator": integrator},
        "forcing": {"const-gravity": {"grav1": -9.8}},
    }


def build(scheme, integrator, tmp):
    from snapy import MeshBlock, MeshBlockOptions
    p = os.path.join(tmp, "deck.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(config(scheme, integrator), f)
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
        for scheme, name in ((1, "hevi-ark2"), (9, "ark-unknown")):
            try:
                build(scheme, name, tmp)
                fails.append("accepted scheme %d with %s" % (scheme, name))
            except Exception as e:  # TORCH_CHECK -> RuntimeError
                print("rejected scheme %d, %s: %s" % (scheme, name, str(e).splitlines()[0][:100]))
    if fails:
        print("FAILED:", "; ".join(fails))
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
