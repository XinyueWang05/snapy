#!/usr/bin/env python3
"""C24 n5 full-sphere metrics, rest, tracer, and conduction (#280)."""
import argparse
import math
import os
import tempfile
from pathlib import Path
import torch
import yaml
import snapy

HERE = Path(__file__).resolve().parent
P0, H0, G = 1.e6, 6.e4, 22.

def setup(arm, implicit, device):
    cfg = yaml.safe_load((HERE / "test_stage2_cubed_stretched.yaml").read_text())
    cfg["integration"]["implicit-scheme"] = 9 if implicit else 0
    if arm == "tracer":
        cfg.pop("forcing")
        cfg["scalar"] = {"nvar": 1, "names": ["uniform_q"],
                         "reconstruct": {"type": "weno5", "scale": True, "shock": False},
                         "riemann-solver": {"type": "upwind"}}
    if arm == "conduction":
        cfg["forcing"]["diffusion"] = {"nu_iso": 0., "kappa_iso": 1.e5,
            "dynamic": False, "temperature_inner": 0., "temperature_outer": 0.,
            "implicit_radial": False}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        name = f.name
    try:
        options = snapy.MeshOptions.from_yaml(name)
        options.blocks_per_process(6)
        options.set_local_horizontal_cells(24, 24)
        mesh = snapy.Mesh(options)
    finally:
        os.unlink(name)
    blocks = list(mesh.blocks)
    assert len(blocks) == 6
    for b in blocks:
        b.to(torch.device(device))
    return mesh, blocks

def initial(blocks, tracer):
    from snapy import kIDN, kIPR, kIV1, kIV2, kIV3
    if not tracer:
        f = blocks[0].module("coord").buffer("x1f").double().cpu()[3:34]
        dr, z = (f[1:] - f[:-1]).contiguous(), (f[1:] + f[:-1]) / 2
        p = P0 * torch.exp(-(z - z[0]) / H0)
        col = torch.zeros(kIPR + 1, 1, 1, 30, dtype=torch.float64)
        col[kIDN, 0, 0], col[kIPR, 0, 0] = p / (G * H0), p
        wb, _, _ = snapy.balance_column(col, dr, G, rtol=1.e-13, stretched=True)
    state = []
    for ib, b in enumerate(blocks):
        w = dict(b.named_buffers())["hydro.D"].clone().zero_()
        if tracer:
            c = b.module("coord")
            beta, alpha = torch.meshgrid(c.buffer("x3v").cpu(),
                                         c.buffer("x2v").cpu(), indexing="ij")
            face = int(b.get_layout().loc_of(ib)[2])
            _, lat = snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face), alpha, beta)
            vel = torch.zeros((3,) + tuple(alpha.shape), dtype=torch.float64)
            vel[2] = 30. * torch.cos(lat)
            snapy.coord.cs_sph_to_contra_(vel, alpha, beta, face)
            q = dict(b.named_buffers())["scalar.D"].clone().zero_()
            w[kIDN], w[kIPR], q[0] = 1., 1.e5, 0.01
            for comp, v in zip((kIV1, kIV2, kIV3), vel):
                w[comp] = v.unsqueeze(-1).to(w)
            state.append({"hydro_w": w, "scalar_r": q})
        else:
            for comp in (kIDN, kIPR):
                w[comp][..., 3:33] = wb[comp, 0, 0].to(w)
                w[comp][..., :3] = w[comp][..., 3:4]
                w[comp][..., 33:] = w[comp][..., 32:33]
            state.append({"hydro_w": w})
    return state

def check(arm, implicit, device, prepared=None):
    mesh, blocks = prepared if prepared is not None else setup(arm, implicit, device)
    if arm == "widths":
        errors, widths = [], []
        for b in blocks:
            c = b.module("coord")
            f, d = c.buffer("x1f"), c.buffer("dx1f")[3:33]
            errors.append((d - (f[4:34] - f[3:33])).abs().max().item())
            widths.extend(d.cpu().tolist())
        assert max(widths) / min(widths) > 4.
        value, tol = max(errors), 1.e-8
    else:
        state, _ = mesh.initialize(initial(blocks, arm == "tracer"))
        stages = len(blocks[0].module("intg").stages)
        for cycle in range(1, 101):
            mesh.set_cycle(cycle)
            dt = mesh.max_time_step(state)
            for stage in range(stages):
                mesh.forward(state, dt, stage)
            assert mesh.check_redo(state) == 0, f"rejected cycle {cycle}"
        value = 0.
        for b, mv in zip(blocks, state):
            sl = b.part((0, 0, 0), False)[1:]
            u = mv["hydro_u"]
            if arm == "tracer":
                error = (mv["scalar_s"][0][sl] / u[snapy.kIDN][sl] - .01).abs().max().item()
            else:
                vel = u[1:4][(slice(None),) + tuple(sl)] / u[snapy.kIDN][sl]
                error = vel.abs().max().item() / math.sqrt(1.4 * G * H0)
            value = max(value, error)
        tol = 2.e-13 if arm == "tracer" else 2.e-12
    assert math.isfinite(value) and value <= tol, f"{arm}: {value} > {tol}"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(125)
    explicit = setup("rest", False, a.device)
    check("widths", False, a.device, explicit)
    check("rest", False, a.device, explicit)
    for arm, implicit in (("rest", True), ("tracer", False), ("conduction", False)):
        check(arm, implicit, a.device)
