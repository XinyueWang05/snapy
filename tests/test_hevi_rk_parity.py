#!/usr/bin/env python3
"""hevi-ark2 against the rk path (implicit-integrator: rk) for the step-end operators the
rk path applies around the dynamics:

  saturation  final-stage saturation adjustment (thermo with reactions). A supersaturated
              moist column at rest (vapor <=> cloud nucleation) runs one step of SAT_DT on each
              path. Both must condense (cloud > 0) and agree: the dynamics of one short step
              at rest is round-off, so max |cloud_hevi - cloud_rk| <= SAT_TOL max cloud_rk.
  solid       solid-cell refill. A moving isentropic column with 4 solid cells at the top
              runs NSTEP steps on each path; on both, the solid cells hold their fill state
              exactly (bitwise) and the fluid mass is conserved to MASS_TOL.

  python test_hevi_rk_parity.py
"""
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

SAT_TOL, MASS_TOL, NSTEP, SAT_DT = 1.0e-6, 1.0e-12, 20, 1.0e-3


def make_block(cfg):
    from snapy import MeshBlock, MeshBlockOptions

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "deck.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(cfg, f)
        b = MeshBlock(MeshBlockOptions.from_yaml(p))
    b.to(torch.device("cpu"), torch.float64)
    return b


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
    return v["hydro_u"][kICY + 1][I].clone(), b.hevi_solve_stats()[0].item()


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
    return exact, mass, bool(torch.isfinite(u).all()), b.hevi_solve_stats()[0].item()


def main():
    torch.set_default_dtype(torch.float64)
    fails = []
    ch, sh = saturation(None)
    cr, sr = saturation("rk")
    diff = (ch - cr).abs().max().item()
    scale = cr.abs().max().item()
    print("saturation: max cloud hevi %.6e, rk %.6e; max |diff| %.3e (%.3e of max), "
          "hevi stages %d, rk-path hevi stages %d" % (ch.max().item(), cr.max().item(), diff,
                                                     diff / max(scale, 1e-300), sh, sr))
    if not (ch.max().item() > 0 and cr.max().item() > 0):
        fails.append("saturation: no condensate on one path")
    elif not diff <= SAT_TOL * scale:
        fails.append("saturation: paths differ by %.3e of max cloud" % (diff / scale))
    if not (sh > 0 and sr == 0):
        fails.append("saturation: integrator selection wrong (hevi %d, rk %d)" % (sh, sr))
    for name, integ in (("hevi-ark2", None), ("rk", "rk")):
        exact, mass, finite, st = solid(integ)
        print("solid %-9s: solid cells bitwise their fill state %s, fluid mass drift %.3e, "
              "finite %s, hevi stages %d" % (name, exact, mass, finite, st))
        if not (exact and finite and mass <= MASS_TOL):
            fails.append("solid %s: exact %s finite %s mass %.3e" % (name, exact, finite, mass))
        if (st > 0) != (integ is None):
            fails.append("solid %s: integrator selection wrong" % name)
    print("FAILED: " + "; ".join(fails) if fails else "PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
