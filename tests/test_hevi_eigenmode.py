#!/usr/bin/env python3
"""The scheme-9 wind mode at the default step dt0: it grows on the rk path and decays under
hevi-ark2 (the default for implicit-scheme 9 with rk3).

The box of the LMARS energy-flux study (fork branch lmars-energy-flux-test 85a3ce1b,
run_arm.py) cut to 16 columns: x1 vertical, 450 km, 100 levels (dz 4.5 km), reflecting;
x2 periodic, NX2 columns of dx 5 km; dry H2-He (cv_R 2.5), isothermal 100 K, g 8.87,
10 bar at the bottom, balanced with snapy.balance_column (rtol 1e-12, 400 sweeps);
weno5, lmars, rk3, cfl 0.3, implicit-scheme 9, uniform wind u = 400 m/s (Mach 0.54).
dt0 = the block's max_time_step (1.313 s here). From noise this box grows on the rk path
with an e-folding time of about 3 h (rms w 5.8e-7 -> 5.2e-5 from 1 h to 8 h), into a
wavenumber-1 mode (lambda = 80 km) with w peaked at about 0.85 of the height.
test_hevi_eigenmode_seed.csv holds that mode: the k = 1 Fourier profile of every primitive
variable of the rk-path state after 8 h from 1e-6 K noise (seed 20261025), scaled to a
k = 1 max|w| of 1e-6 m/s. The test adds it to the balanced column, runs HOURS at dt0
and measures the wavenumber-1 amplitude of w (max over z) at the start and the end.
(An analytic w-only seed of the same shape decays on the rk path: it projects mostly
onto damped modes.)

  python test_hevi_eigenmode.py               the default (hevi-ark2): must not grow,
                                              end/start <= DECAY
  python test_hevi_eigenmode.py --integrator rk   the opt-out rk path: must grow,
                                              end/start >= GROW (the RED side)

Exit 0 when the arm behaves as stated, 1 otherwise. On a tree without hevi-ark2 (the
base), the default arm is the rk path and fails: that is the RED run.
"""
import argparse
import math
import os
import sys
import tempfile
import time

import torch
import yaml

G, T0, ZT, PS = 8.87, 100.0, 450.0e3, 1.0e6
NX1, NX2, DX, U = 100, 16, 5.0e3, 400.0
DECAY, GROW = 1.0, 2.0


def config(integrator):
    integ = {"type": "rk3", "cfl": 0.3, "implicit-scheme": 9, "nlim": -1, "tlim": 1.e9}
    if integrator:
        integ["implicit-integrator"] = integrator
    return {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry", "composition": {"H": 1.5, "He": 0.15}, "cv_R": 2.5}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": ZT, "x2min": 0., "x2max": NX2 * DX,
                                "x3min": -0.5, "x3max": 0.5},
                     "cells": {"nx1": NX1, "nx2": NX2, "nx3": 1, "nghost": 3}},
        "dynamics": {"equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                           "pressure-floor": 1.e-10, "limiter": True,
                                           "max-iter": 10, "ftol": 1.e-6},
                     "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": True},
                                     "horizontal": {"type": "weno5", "scale": True,
                                                    "shock": True}},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": integ,
        "forcing": {"const-gravity": {"grav1": -G}},
    }


def k1_amplitude(w, ng):
    ww = w[0, ng:ng + NX2, ng:ng + NX1]  # (x2, x1)
    return (2.0 / NX2 * torch.fft.rfft(ww, dim=0)[1].abs()).max().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--integrator", default=None, help="implicit-integrator key (default: none)")
    p.add_argument("--hours", type=float, default=3.0)
    a = p.parse_args()

    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "deck.yaml")
        with open(f, "w") as fh:
            yaml.safe_dump(config(a.integrator), fh)
        b = MeshBlock(MeshBlockOptions.from_yaml(f))
    b.to(torch.device("cpu"), torch.float64)
    eos = b.module("hydro.eos")
    ng = 3
    x1v, dx1f = b.buffer("coord.x1v"), b.buffer("coord.dx1f")
    il, iu = ng, ng + NX1
    w = b.buffer("hydro.D").clone().zero_()
    probe = w.clone()
    probe[kIDN], probe[kIPR] = 1.0, 1.0e5
    rd = 1.0e5 / eos.compute("W->T", [probe]).flatten()[0].item()
    z = x1v[il:iu]
    pcol = PS * torch.exp(-G * (z - z[0]) / (rd * T0))
    wi = torch.zeros(w.shape[0], 1, w.shape[2], NX1)
    wi[kIPR], wi[kIDN] = pcol, pcol / (rd * T0)
    wb, _, _ = snapy.balance_column(wi.contiguous(), dx1f[il:iu].contiguous(), G, True,
                                    1.0e-12, 400, "cartesian")
    w[..., il:iu] = wb
    w[..., :il], w[..., iu:] = wb[..., :1], wb[..., -1:]
    w[kIV1 + 1] = U
    seed = torch.zeros(5, NX1, dtype=torch.complex128)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "test_hevi_eigenmode_seed.csv")) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("var"):
                continue
            var, lev, re, im = line.split(",")
            seed[int(var), int(lev)] = complex(float(re), float(im))
    phase = torch.exp(2j * math.pi * torch.arange(NX2, dtype=torch.float64) / NX2)
    dq = (2.0 / NX2) * (seed.unsqueeze(1) * phase.view(1, -1, 1)).real  # (var, x2, x1)
    w[:, 0, ng:ng + NX2, il:iu] += dq
    v, _ = b.initialize({"hydro_w": w})

    dt0 = b.max_time_step(v)
    a_start = k1_amplitude(v["hydro_w"][kIV1], ng)
    nstep = int(round(a.hours * 3600.0 / dt0))
    w0 = time.time()
    finite = True
    for _ in range(nstep):
        for s in range(len(b.module("intg").stages)):
            b.forward(v, dt0, s)
        if b.check_redo(v) != 0:
            finite = False
            break
    finite = finite and bool(torch.isfinite(v["hydro_u"]).all())
    a_end = k1_amplitude(v["hydro_w"][kIV1], ng) if finite else float("inf")
    ratio = a_end / a_start
    stats = b.hevi_solve_stats().tolist() if hasattr(b, "hevi_solve_stats") else None
    hevi = bool(stats and stats[0] > 0)
    print("integrator key %s -> %s: dt0 %.4f s, %d steps (%.2f h), k=1 |w| %.3e -> %.3e "
          "(ratio %.3f), wall %.0f s, solve stats %s"
          % (a.integrator, "hevi-ark2" if hevi else "rk", dt0, nstep, nstep * dt0 / 3600.0,
             a_start, a_end, ratio, time.time() - w0, stats))
    if a.integrator == "rk":
        ok = finite and ratio >= GROW
        print("PASSED (rk path grows)" if ok else "FAILED: rk path did not grow (ratio %.3f)" % ratio)
    else:
        ok = finite and hevi and ratio <= DECAY
        print("PASSED (hevi-ark2 does not grow)" if ok else
              "FAILED: default arm %s, ratio %.3f (> %.1f means it grows)"
              % ("is hevi-ark2" if hevi else "is the rk path", ratio, DECAY))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
