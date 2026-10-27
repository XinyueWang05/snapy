#!/usr/bin/env python3
"""Scheme-1 design checks on ba2b28ce (Xi, ts 1793068432); driver derived from
scheme1_donor_momentum/run_arm.py (same deck and diagnostics). Additions:
  --shear       u(z) = u0 * z / 450 km at cell centres (flat column with --uniform-x2)
  --moist       dry + vapor + cloud deck of tests/test_tracer_dry_convention.yaml
                (32 km / 32 levels, g 10, 320 K, vapour 0.02, cloud 0.005)
  --prof-days   dump the horizontal-mean m_t (= m2) and u profiles at these days
Velocities use the TOTAL density (dry + every species), as the EOS does.

Original docstring follows.
One arm of the LMARS strong-wind study (thread ts 1792975246 / 1792995379).

Deck rebuilt from the thread spec (the lmars-energy-flux-test branch was not
on the fork): x1 vertical 450 km / 100 levels, reflecting walls; x2 periodic;
dry H2-He (H 1.5, He 0.15, cv_R 2.5); g 8.87; isothermal T0 100 K, 10 bar
bottom, discretely balanced with snapy.balance_column; weno5 both directions;
rk3, cfl 0.3; uniform wind u0 in x2; isobaric 1e-6 K uniform-random T noise.

Writes <out>.csv (diagnostic rows) and <out>.json (summary). Diagnostics:
rms w, max|u-u0|/u0, global sums of rho, m1, m2, m3, E and E + rho g z.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import yaml

G, T0, PS, LZ, NZ = 8.87, 100.0, 1.0e6, 450.0e3, 100


def deck(args):
    return {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry", "composition": {"H": 1.5, "He": 0.15}, "cv_R": 2.5}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": LZ, "x2min": 0.0, "x2max": args.nx2 * args.dx,
                                "x3min": 0.0, "x3max": args.dx},
                     "cells": {"nx1": NZ, "nx2": args.nx2, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                  "pressure-floor": 1.e-10, "limiter": True,
                                  "max-iter": 10, "ftol": 1.e-6},
            "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": True},
                            "horizontal": {"type": "weno5", "scale": True, "shock": True}},
            "riemann-solver": {"type": args.riemann}},
        "boundary-condition": {"external": {
            "x1-inner": "reflecting", "x1-outer": "reflecting",
            "x2-inner": "periodic", "x2-outer": "periodic",
            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": args.cfl, "implicit-scheme": args.scheme,
                        "nlim": -1, "tlim": args.days * 86400.0, "ncycle_out": 1000000},
        "forcing": {"const-gravity": {"grav1": -G}},
    }


def moist_deck(args):
    cfg = deck(args)
    cfg["species"] = [
        {"name": "dry", "composition": {"H": 1.6667, "He": 0.16667}, "cv_R": 2.5},
        {"name": "vapor", "composition": {"H": 2, "O": 1}, "cv_R": 2.5, "u0_R": 0.0},
        {"name": "cloud", "composition": {"H": 2, "O": 1}, "cv_R": 9.0, "u0_R": -3430.0}]
    cfg["reactions"] = [{"equation": "vapor <=> cloud", "type": "nucleation",
                         "rate-constant": {"formula": "h2o_ideal"}}]
    cfg["geometry"]["bounds"]["x1max"] = MLZ
    cfg["geometry"]["cells"]["nx1"] = MNZ
    cfg["forcing"]["const-gravity"]["grav1"] = -MG
    return cfg


MLZ, MNZ, MG, MT0, MPS, MRD = 3.2e4, 32, 10.0, 320.0, 1.0e5, 3900.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dx", type=float, default=100.e3)
    ap.add_argument("--nx2", type=int, default=64)
    ap.add_argument("--u0", type=float, default=0.0)
    ap.add_argument("--scheme", type=int, default=0)
    ap.add_argument("--riemann", default="lmars")
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--noise", type=float, default=1.e-6)
    ap.add_argument("--uniform-x2", action="store_true",
                    help="same noise in every x2 column: an exactly 1-D column whose dt keeps the x2 limit")
    ap.add_argument("--shear", action="store_true")
    ap.add_argument("--moist", action="store_true")
    ap.add_argument("--prof-days", type=float, nargs="*", default=[1.0, 2.0, 3.0])
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--diag-dt", type=float, default=600.0)
    ap.add_argument("--wmax-stop", type=float, default=200.0, help="stop as blown up above this max|w| (m/s)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1, kIV2, kIV3

    global G, T0, PS, LZ, NZ
    if args.moist:
        G, T0, PS, LZ, NZ = MG, MT0, MPS, MLZ, MNZ
    cfg = moist_deck(args) if args.moist else deck(args)
    with open(args.out + ".yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    block = MeshBlock(MeshBlockOptions.from_yaml(args.out + ".yaml"))
    block.to(torch.device("cpu"))

    ng, nx2 = 3, args.nx2
    dz = LZ / NZ
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    rgas = 8.31446261815324
    mu = 1.5 * 1.00794e-3 + 0.15 * 4.002602e-3
    rd = MRD if args.moist else rgas / mu
    z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * dz
    p = PS * torch.exp(-G * z / (rd * T0))
    col = torch.zeros(kIPR + 1, 1, 1, NZ, dtype=torch.float64)
    col[kIDN, 0, 0] = p / (rd * T0)
    col[kIPR, 0, 0] = p
    if args.moist:  # continuous hydrostatic profile (balance_column is for the dry deck)
        wb, err = col, 0.0
    else:
        wb, err, _ = snapy.balance_column(col, torch.full((NZ,), dz, dtype=torch.float64), G)
    jl = ng if nx2 > 1 else 0
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ] = wb[c, 0, 0].to(w)
    gen = torch.Generator().manual_seed(args.seed)
    if args.noise > 0:
        dT = (2.0 * torch.rand(w.shape[1], 1 if args.uniform_x2 else nx2, NZ, generator=gen) - 1.0) * args.noise
        sl = w[kIDN][:, jl:jl + nx2, ng:ng + NZ]
        w[kIDN][:, jl:jl + nx2, ng:ng + NZ] = sl / (1.0 + dT / T0)  # isobaric
    for c in (kIDN, kIPR):
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ:] = w[c][..., ng + NZ - 1:ng + NZ]
    if args.moist:
        from snapy import kICY
        w[kICY] = 0.02     # vapour mass fraction of the total
        w[kICY + 1] = 0.005  # cloud mass fraction of the total
    zfull = (torch.arange(w.shape[-1], dtype=torch.float64) - ng + 0.5) * dz
    uprof = args.u0 * zfull / LZ if args.shear else torch.full_like(zfull, args.u0)
    w[kIV2] = uprof.view(1, 1, -1)
    uref = uprof[ng:ng + NZ].view(1, 1, NZ)
    bv, _ = block.initialize({"hydro_w": w})

    interior = (slice(None), slice(0, 1), slice(jl, jl + nx2), slice(ng, ng + NZ))
    zc = z.view(1, 1, NZ)
    vol = args.dx * dz * args.dx  # x3 width = dx (uniform cells)

    def diag(t):
        u = bv["hydro_u"][interior]
        rho = u[kIDN] + (u[5:].sum(0) if u.shape[0] > 5 else 0.0)  # total density
        wv = u[kIV1] / rho
        uv = u[kIV2] / rho
        rec = {"t": t, "rms_w": wv.pow(2).mean().sqrt().item(), "max_w": wv.abs().max().item()}
        rec["max_du_rel"] = ((uv - uref).abs().max() / args.u0).item() if args.u0 != 0 else (uv.abs().max()).item()
        rec["rms_du"] = (uv - uref).pow(2).mean().sqrt().item()
        rec["S_rho"] = rho.sum().item() * vol
        rec["S_m1"] = u[kIV1].sum().item() * vol
        rec["S_m2"] = u[kIV2].sum().item() * vol
        rec["S_m3"] = u[kIV3].sum().item() * vol
        rec["S_E"] = u[kIPR].sum().item() * vol
        rec["S_Etot"] = (u[kIPR] + rho * G * zc).sum().item() * vol
        rec["KE_pert"] = (0.5 * rho * ((uv - uref).pow(2) + wv.pow(2))).sum().item() * vol
        return rec, wv

    gamma = 1.4

    def dominant(wv):
        """Dominant x2 wavelength of w in units of dx (power summed over levels)."""
        if nx2 < 2 or not torch.isfinite(wv).all():
            return None
        spec = np.abs(np.fft.rfft(wv[0].numpy(), axis=0)) ** 2  # (nx2//2+1, NZ)
        pk = spec.sum(axis=1)
        pk[0] = 0.0
        k = int(np.argmax(pk))
        return nx2 / k if k > 0 else None

    cs = math.sqrt(gamma * rd * T0)
    nstage = len(block.module("intg").stages)
    out_csv = open(args.out + ".csv", "w")
    rec0, _ = diag(0.0)
    cols = ["t", "step", "dt"] + [k for k in rec0 if k != "t"]
    out_csv.write(",".join(cols) + "\n")

    last = {"t": None}

    def write(rec, step, dt):
        if rec["t"] == last["t"]:
            return
        last["t"] = rec["t"]
        out_csv.write(",".join(repr(x) for x in [rec["t"], step, dt] + [rec[k] for k in cols[3:]]) + "\n")
        out_csv.flush()

    write(rec0, 0, 0.0)
    t, step, redos, status = 0.0, 0, 0, "finished"
    tlim = args.days * 86400.0
    next_diag = args.diag_dt
    dts = []
    wall0 = time.time()
    lam_mid = None
    prof_t = sorted(d * 86400.0 for d in args.prof_days)
    prof = open(args.out + ".prof.csv", "w")
    prof.write("t,k,z,m2_mean,u_mean,u_init\n")

    def dump_prof(tt):
        u = bv["hydro_u"][interior]
        rho = u[kIDN] + (u[5:].sum(0) if u.shape[0] > 5 else 0.0)
        m2 = u[kIV2].mean(dim=(0, 1))
        um = (u[kIV2] / rho).mean(dim=(0, 1))
        for k in range(NZ):
            prof.write("%r,%d,%r,%r,%r,%r\n" % (tt, k, z[k].item(), m2[k].item(), um[k].item(), uref[0, 0, k].item()))
        prof.flush()

    dump_prof(0.0)
    while t < tlim:
        dt = block.max_time_step(bv)
        dt = min(dt, tlim - t)
        for s in range(nstage):
            block.forward(bv, dt, s)
        e = block.check_redo(bv)
        if e > 0:
            redos += 1
            if redos > 50:
                status = "blew_up(redo>50)"
                break
            continue
        if e < 0:
            status = "blew_up(terminate)"
            break
        t += dt
        step += 1
        dts.append(dt)
        while prof_t and t >= prof_t[0] - 1e-6:
            dump_prof(t)
            prof_t.pop(0)
        if t >= next_diag - 1e-9 or t >= tlim:
            rec, wv = diag(t)
            write(rec, step, dt)
            if lam_mid is None and rec["rms_w"] >= 1.e-2:
                lam_mid = (t, dominant(wv))
            next_diag += args.diag_dt
            if not math.isfinite(rec["rms_w"]):
                status = "blew_up(nan)"
                break
            if rec["max_w"] > args.wmax_stop:
                status = "blew_up(max_w>%g)" % args.wmax_stop
                break
    rec, wv = diag(t)
    write(rec, step, dts[-1] if dts else 0.0)
    out_csv.close()
    lam_dx = dominant(wv)
    import hashlib
    ufin = bv["hydro_u"].detach().cpu().contiguous().numpy()
    np.save(args.out + ".final_u.npy", ufin)
    state_sha256 = hashlib.sha256(ufin.tobytes()).hexdigest()
    summ = dict(vars(args))
    summ.update(status=status, t_end=t, t_end_days=t / 86400.0, steps=step, redos=redos,
                dt_min=min(dts) if dts else None, dt_max=max(dts) if dts else None,
                dt_median=float(np.median(dts)) if dts else None, cs=cs,
                mach=args.u0 / cs, hcour=args.u0 * (float(np.median(dts)) if dts else 0) / args.dx,
                vcour=cs * (float(np.median(dts)) if dts else 0) / dz, dz=dz, rd=rd,
                balance_err=float(err) if not torch.is_tensor(err) else float(err.max()),
                lam_dx_end=lam_dx, lam_dx_mid=lam_mid, wall_s=time.time() - wall0,
                snapy_version=getattr(snapy, "__version__", "?"), torch_version=torch.__version__,
                snapy_file=snapy.__file__, state_sha256=state_sha256)
    with open(args.out + ".json", "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    sys.exit(main())
