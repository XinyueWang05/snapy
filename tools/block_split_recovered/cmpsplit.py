"""nb2 vs nb1 at 1 rank per config dir: per-variable max|u2-u1|/max|u1| (rows of hydro_u: 0 rho/phi, 1 m1, 2 m2, 3 m3, 4 E),
max over variables, dt sequence equal; plus optional cross-build comparisons at equal nb (bitwise flags)."""
import sys, torch
def load(d, case, nb):
    r = torch.load(f"{d}/w1_nb{nb}/{case}_w1_nb{nb}_r0.pt"); f = r["fields"]; panels = []
    for p in range(6):
        ks = [k for k in f if k[0] == p]; x2s = sorted({k[1] for k in ks}); x3s = sorted({k[2] for k in ks})
        panels.append(torch.cat([torch.cat([f[(p, a, b)] for a in x2s], dim=2) for b in x3s], dim=1))
    return torch.stack(panels), r["dt"], r["wall"], r["drift"]
cases = ["f_slow", "f_deck", "g_slow", "g_deck", "h_slow", "h_deck", "w2_1", "w2_4"]
root = sys.argv[1]; cfgs = sys.argv[2:]
for cfg in cfgs:
    print(f"== {cfg}")
    for c in cases:
        try: a, da, wa, _ = load(f"{root}/{cfg}", c, 1); b, db, wb, _ = load(f"{root}/{cfg}", c, 2)
        except FileNotFoundError: print(f"  {c:7} missing"); continue
        per = [((b[:, v] - a[:, v]).abs().max() / a[:, v].abs().max().clamp_min(1e-300)).item() for v in range(a.shape[1])]
        print(f"  {c:7} max {max(per):.3e}  per-var [{', '.join(f'{x:.1e}' for x in per)}]  dt same {torch.equal(da, db)}  wall nb1/nb2 {wa:.2f}/{wb:.2f}s")
