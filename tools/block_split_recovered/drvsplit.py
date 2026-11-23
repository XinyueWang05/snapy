"""#365 closure, multi-rank: arm (l) of tests/test_axial_momentum_flux_form.py (pinned checkout), one case per call,
under torchrun (or plain python for 1 rank). Config exactly as arm (l): config("full") + coriolis + arm edit +
interp_order 4 + conserve-axial-momentum-closure true; W2: sw_config("full", m Omega), sw_state(.., u0/m).
Only change for ranks: blocks_per_process = 6 nb^2 / WORLD_SIZE (nb = NB env, default 1, sets nb2 = nb3).
The global AAM is summed over ranks with a separate Python gloo group (file rendezvous). Saves per rank: interior hydro_u
of every local block keyed by (panel, j0, k0), the per-step global L and dt, wall time.
usage: drv365.py <testfile> <case> <nstep> <outdir>; case in w2_1 w2_4 f_slow f_deck g_slow g_deck h_slow h_deck"""
import importlib.util, math, os, sys, tempfile, time
import torch, yaml
import torch.distributed as dist
tf, case, nstep, outd = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
spec = importlib.util.spec_from_file_location("t", tf); t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
MODE = os.environ.get("MODE", "full"); NB = int(os.environ.get("NB", "1")); nblk = 6 * NB * NB; assert nblk % world == 0, (nblk, world)
outd = os.path.abspath(outd); os.makedirs(outd, exist_ok=True)
if world > 1:
    dist.init_process_group("gloo", init_method="file://" + os.path.join(outd, f"pg_{case}"), rank=rank, world_size=world)
def gsum(x):
    if world == 1: return x
    v = torch.tensor([x], dtype=torch.float64); dist.all_reduce(v); return v.item()
def build(cfg):
    from snapy import Mesh, MeshOptions
    nx = cfg["geometry"]["cells"]["nx2"]
    cfg["distribute"]["nb2"] = cfg["distribute"]["nb3"] = NB
    cfg["distribute"]["blocks_per_process"] = nblk // world
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=outd) as f:
        yaml.safe_dump(cfg, f); tmp = f.name
    try:
        o = MeshOptions.from_yaml(tmp); o.blocks_per_process(nblk // world); o.set_local_horizontal_cells(nx // NB, nx // NB)
        mesh = Mesh(o)
    finally:
        os.unlink(tmp)
    blocks = list(mesh.blocks)
    for b in blocks: b.to(torch.device("cpu"))
    assert len(blocks) == nblk // world
    return mesh, blocks
base = yaml.safe_load(open(os.path.join(os.path.dirname(tf), "test_axial_momentum_flux_form.yaml")))
def with_full(cfg):
    cfg["geometry"]["cells"]["interp_order"] = 4
    if os.environ.get("CLOSURE") is not None:  # unset: no key at all (main has none)
        cfg["dynamics"]["conserve-axial-momentum-closure"] = os.environ["CLOSURE"] == "1"
    return cfg
def jet(lon, lat, rc):
    return t.W_SB * rc + t.U_PERT * torch.cos(lat) ** 2 * torch.sin(2.0 * lon)
if case.startswith("w2"):
    m = float(case[3:]); om = m * t.SW_OMEGA
    mesh, blocks = build(with_full(t.sw_config(MODE, om)))
    mv = t.sw_state(blocks, om, t.SW_U0 / m)
else:
    tag, rate = case.split("_"); om = {"slow": t.W_SB, "deck": 7.0e-5}[rate]
    cfg = yaml.safe_load(yaml.safe_dump(base)); cfg["dynamics"]["conserve-axial-momentum"] = MODE
    cfg["forcing"]["coriolis"] = {"type": "xyz", "omega1": om}
    state = {}
    if tag == "g":
        cfg["forcing"]["coriolis"]["traditional"] = True
        cfg["geometry"]["bounds"]["x1max"] = float(cfg["geometry"]["bounds"]["x1min"]) + 8.0
        cfg["integration"]["implicit-scheme"] = 9
    elif tag == "h":
        cfg["integration"]["implicit-scheme"] = 9; state = {"w_up": t.W_UP}
    mesh, blocks = build(with_full(cfg))
    mv = t.initial_state(cfg, blocks, jet, True, om, **state)
mv, _ = mesh.initialize(mv)
intg = blocks[0].module("intg")
L0 = gsum(t.aam(blocks, mv, om)); Ls, dts = [L0], []; drift = 0.0
t0 = time.perf_counter()
for n in range(1, nstep + 1):
    mesh.set_cycle(n); dt = mesh.max_time_step(mv); dts.append(dt)
    for s in range(len(intg.stages)): mesh.forward(mv, dt, s)
    if not case.startswith("w2"): assert mesh.check_redo(mv) == 0
    L = gsum(t.aam(blocks, mv, om)); Ls.append(L)
    d = abs(L - L0) / abs(L0); drift = max(drift, d) if math.isfinite(d) else math.inf
wall = time.perf_counter() - t0
fields = {}
for ib, b in enumerate(blocks):
    c = b.module("coord"); s3, s2, s1 = t.interior(b)
    lon = c.buffer("longitude").double()[s3, s2].reshape(-1)[0]; lat = c.buffer("latitude").double()[s3, s2].reshape(-1)[0]
    v = torch.stack([torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)])
    ax = int(v.abs().argmax()); panel = 2 * ax + (0 if v[ax] > 0 else 1)
    x2 = c.buffer("x2v").double()[s2][0].item(); x3 = c.buffer("x3v").double()[s3][0].item()
    fields[(panel, round(x2, 9), round(x3, 9))] = mv[ib]["hydro_u"].double()[(slice(None), s3, s2, s1)].clone()
torch.save(dict(case=case, world=world, rank=rank, NB=NB, drift=drift, L=torch.tensor(Ls), dt=torch.tensor(dts), wall=wall, fields=fields),
           os.path.join(outd, f"{case}_w{world}_nb{NB}_r{rank}.pt"))
if rank == 0:
    print(f"RESULT mode={MODE} closure={os.environ.get('CLOSURE')} case={case} world={world} NB={NB} steps={nstep} drift={drift:.3e} wall={wall:.2f}s", flush=True)
if world > 1: dist.destroy_process_group()
