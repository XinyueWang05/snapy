#!/usr/bin/env python3
"""One-step spectral radius of the #288 column (the deck of test_implicit_gravity_tall_column.py as
one column: nz 45, rk3 + weno5 + lmars, vic-full, corrected-PE work on) at N dt = 150, by central
differences in (rho, u1, p). The wall slopes' third points must be implicit: radius <= 1 + 1e-6."""
import math, os, sys, tempfile
import numpy as np, torch, yaml
import snapy
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1
import test_implicit_gravity_tall_column as tall

dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
if dev == "cuda" and not torch.cuda.is_available():
    sys.exit(125)
cfg = tall.config()
cfg["geometry"]["cells"]["nx2"] = 1
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
    yaml.safe_dump(cfg, f)
block = MeshBlock(MeshBlockOptions.from_yaml(f.name))
os.unlink(f.name)
block.to(torch.device(dev))
NZ, NG, G, RT, GAMMA = tall.NZ, 3, tall.GRAV, tall.RD * tall.T0, tall.GAMMA
z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * tall.DZ
col = torch.zeros(kIPR + 1, 1, 1, NZ, dtype=torch.float64)
col[kIPR, 0, 0] = tall.PS * torch.exp(-G * z / RT)
col[kIDN, 0, 0] = col[kIPR, 0, 0] / RT
wb = snapy.balance_column(col, torch.full((NZ,), tall.DZ, dtype=torch.float64), G)[0][:, 0, 0]
base = dict(block.named_buffers())["hydro.D"].clone().zero_()
for c in (kIDN, kIPR):
    base[c] = torch.cat([wb[c, :1].expand(NG), wb[c], wb[c, -1:].expand(NG)]).to(base)
dt = 150.0 / math.sqrt((GAMMA - 1) / GAMMA * G * G / RT)
scale = torch.stack([wb[kIDN], (GAMMA * wb[kIPR] / wb[kIDN]).sqrt(), wb[kIPR]], 1).reshape(-1)
nstage = len(block.module("intg").stages)


def step(w):
    v, _ = block.initialize({"hydro_w": w})
    for s in range(nstage):
        block.forward(v, dt, s)
    u = v["hydro_u"][..., NG:NG + NZ].reshape(5, NZ).cpu()
    p = (GAMMA - 1) * (u[kIPR] - 0.5 * (u[1:4] ** 2).sum(0) / u[kIDN])
    return torch.stack([u[kIDN], u[kIV1] / u[kIDN], p], 1).reshape(-1) / scale


J = np.zeros((3 * NZ, 3 * NZ))
for j in range(3 * NZ):
    out = []
    for sign in (1, -1):
        w = base.clone()
        w[(kIDN, kIV1, kIPR)[j % 3]][..., NG + j // 3] += sign * 1e-6 * scale[j].item()
        out.append(step(w))
    J[:, j] = ((out[0] - out[1]) / 2e-6).numpy()
radius = float(np.abs(np.linalg.eigvals(J)).max())
print(f"#288 column nz {NZ} ({dev}), N dt 150: one-step radius {radius:.10f} (limit 1 + 1e-6)")
sys.exit(0 if radius <= 1 + 1e-6 else 1)
