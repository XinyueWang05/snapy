# snapy#280 route A, stage A — the per-face coefficient layout

Branch `feat/280-stage1-piece2-route-a` on `happysky19/snapy`, cut from
`chengcli/snapy` main `2d9fa419be56513159c695267b68753d5b223b31`.
Every file:line below is read at that base commit unless it names a file this
branch adds.

Owed to **Xinyue** (§1–§4, §6) and to **Jiheng** (§5).

---

## 1. Face and window convention

Line numbers below are at the base commit `2d9fa419` unless the file is one
this branch adds.

Three facts from the shipped code, in order:

* `src/coord/cartesian.cpp:26-27` — face `k` is the **lower** face of cell `k`,
  so a block with `nc1` cells has `nc1 + 1` faces.
* `src/recon/interpolation.hpp:64-65` — `forward` makes the output smaller than
  the input by `stencils() - 1 = 4` along `dim`, so the output has `nc1 - 4`
  elements. Output element `m` is the five-cell window `m .. m+4`, bounded by
  the six faces `m .. m+5`.
* `src/recon/reconstruct.cpp:54-55` —

  ```cpp
  auto outl = wlr[IRT].slice(dim, il - 1, iu + 1);
  auto outr = wlr[ILT].slice(dim, il,     iu + 2);
  ```

  with `il = 3`, `iu = nc1 - 3` for a 5-point stencil
  (`src/recon/reconstruct.cpp:86-88`). Both slices are handed to the **same**
  `forward` call, so output element `m` lands at `wlr[IRT][il-1+m] = wlr[IRT][m+2]`
  and at `wlr[ILT][il+m] = wlr[ILT][m+3]`.

So the two sides of one output element target **two different faces built from
the same five cells** — the lower and the upper face of the window's central
cell `m+2`:

```
row m of cm  ->  target face m+2   (the IRT state, written by left())
row m of cp  ->  target face m+3   (the ILT state, written by right())
```

On a uniform grid `cp = cm.flip`, which hides the distinction completely: a
face-index off-by-one is invisible to every uniform test. This is the
`reconstruct.cpp:54-55` hazard the brief flags, and it is checked explicitly on
a **non-uniform** face vector by `X1Coeffs.polynomial_exactness_and_face_index`
(degree-4 data; each row is required to be closer to its own target face than
to the other one, which fails by `O(h)` under a swap or a shift of one).

The builders take the face vector and nothing else:

```cpp
std::pair<torch::Tensor, torch::Tensor> weno5_x1_coeffs(torch::Tensor const& x1f);
std::pair<torch::Tensor, torch::Tensor> cp5_x1_coeffs (torch::Tensor const& x1f);
```

Both return `{cm, cp}` with `x1f.numel() - 5` rows, float64, on the CPU, for an
`x1f` of at least 6 strictly increasing finite entries.

---

## 2. The WENO5 row: 48 columns

Columns 0–44 are the nine 5-entry rows the kernel walks at
`src/recon/interp_impl.h:77-85`, **in that order**:

| columns | row | meaning |
|---|---|---|
| 0–4   | c1 | candidate polynomial `p_0` (linear weight `d_0`) |
| 5–9   | c2 | candidate polynomial `p_1` (linear weight `d_1`) |
| 10–14 | c3 | candidate polynomial `p_2` (linear weight `d_2`) |
| 15–19 | c4 | `A_0` ┐ |
| 20–24 | c5 | `B_0` ┘ `beta_0 = 13/12 (A_0·u)^2 + 1/4 (B_0·u)^2` |
| 25–29 | c6 | `A_1` ┐ |
| 30–34 | c7 | `B_1` ┘ `beta_1` |
| 35–39 | c8 | `A_2` ┐ |
| 40–44 | c9 | `B_2` ┘ `beta_2` |
| 45 | `d_0` | ┐ |
| 46 | `d_1` | │ the three linear weights |
| 47 | `d_2` | ┘ |

Within a 5-entry row the index is the **window cell offset** `n = 0..4`: the
coefficient multiplies the cell average of block cell `m + n`. Entries outside
a candidate's three-cell stencil are exactly `0.`.

**Stencil order.** `r = 0, 1, 2` are the three-cell stencils in the order the
shipped rows use, largest linear weight first. For the `cp` side (target
`m+3`, the upper face of the central cell) stencil `r` starts at window cell
`k0[r]` with `k0 = {2, 1, 0}`; on a uniform grid that gives
`d = (0.3, 0.6, 0.1)`, exactly the literals at
`src/recon/interp_impl.h:115-117`. The `cm` side is the mirror image of the
same construction (reflect the window, build, then reverse each 5-entry row),
so `cm` and `cp` are each other's mirror by construction and reduce to
`cp = cm.flip` on a uniform window.

### Why 48 and not the 45 Xinyue expected

45 is the **shipped** coefficient count: `kWeno5Coeff` at
`src/recon/interp_impl.cuh:221`, which is the 9 × 5 table and nothing else.
The shipped kernel does not need any more, because on a uniform grid the three
optimal linear weights are the constants `.3 / .6 / .1` hard-coded at
`src/recon/interp_impl.h:115-117`.

On a stretched grid those three weights are **face dependent**: the
fifth-order combination of the three quadratic candidates is only fifth order
if `d_r` is solved from the local face positions. Route A therefore carries
them in the same row, `45 + 3 = 48`, as the design note §4.2 "Storage" says.
In the normalised coordinate `zeta = (x - x_target)/h_central` with
`a = -z_0`, `b = -z_1`, `c = z_4`, `e = z_5` (all positive lengths):

```
d_0 = a b / ((a+e)(b+e))
d_1 = a e (a+b+c+e) / ((a+e)(b+e)(a+c))
d_2 = c e / ((a+c)(a+e))
```

`d_0 + d_1 + d_2 = 1` is an algebraic identity, and every factor is a positive
length, so `d_r > 0` on **any** admissible grid — no negative-weight splitting
is needed. Measured over `q` in `[1/16, 16]`: `min d_r = 5.566198e-08 > 0`,
`max |sum_r d_r - 1| = 4.441e-16`.

### Are the smoothness rows pre-scaled?

**Yes.** `c4 … c9` are scaled so that the kernel's own `13/12` and `1/4`
multipliers (`src/recon/interp_impl.h:108-113`) reproduce the Jiang & Shu
indicator unchanged. Concretely, for the candidate quadratic
`q(zeta) = a_0 + a_1 zeta + a_2 zeta^2` of stencil `r`, written in the
normalised coordinate above (the central cell is `[-1, 0]` for the `cp` side
and has unit width):

```
A_r · u = 2 a_2
B_r · u = ± (2 a_2 - 2 a_1)     (sign per stencil: +, +, -; beta is invariant
                                 under it, the sign only matches the shipped
                                 literals entry for entry)
```

so

```
13/12 (A_r·u)^2 + 1/4 (B_r·u)^2 = a_1^2 - 2 a_1 a_2 + 16/3 a_2^2
                                = sum_{m=1,2} h^{2m-1} int_{I_c} (q^{(m)})^2 dx
```

which is `beta_r` exactly. Because the whole row is built in the normalised
coordinate, **`beta` is dimensionless on any grid**: it does not rescale with
the cell width, so the shipped absolute `eps = 1e-6`
(`src/recon/interp_impl.h:115-117`) keeps its uniform-grid meaning. The
kernel's own arithmetic -- the `13/12` and `1/4` in `beta`, the squaring, the
`eps`, the normalisation -- is unchanged; only the three literal linear
weights are replaced by columns 45-47. Checked by `X1Coeffs.beta_is_scale_free`.

A uniform face vector reproduces the shipped literals
(`src/recon/weno5.cpp:11-23`) to `max |c_new - c_shipped| = 4.441e-16` at
`h = 1` and `h = 7500`, and `1.954e-14` at `h = 1e-3` — **close but not
bitwise**. That difference never runs: a uniform deck keeps the shipped path,
which this branch does not touch.

---

## 3. The cp5 row: 5 columns

One row of `(nx1+1 - 5, 5)` per side, same face convention: `cm` row `m`
targets face `m+2`, `cp` row `m` targets face `m+3`. Column `n = 0..4` is the
coefficient of window cell `m+n`, i.e. the row is the degree-4 (five-cell)
reconstruction at the target face, replacing the shipped constants
`{-1/20, 9/20, 47/60, -13/60, 1/30}` at `src/recon/cp5.cpp:14-16`.

It is **built as `sum_r d_r p_r`** from the WENO5 row (note §4.2), not from an
independent six-node primitive function: that identity is what makes WENO5
fifth order, and the combined form loses fewer digits on a sharp width jump.
The two routes are checked against each other in
`X1Coeffs.cp5_row_matches_an_independent_derivation`:
`max |sum_r d_r p_r − 6-node primitive row| = 1.47148e-15` over every row of
every test grid.

---

## 4. Can `snapy.Reconstruct` be driven with per-face rows supplied from Python?

**Stage A: the rows can be built and installed from Python, and read back, but
they do not yet change what `Reconstruct.forward` computes.**

Plainly:

* `snapy.weno5_x1_coeffs(x1f)` and `snapy.cp5_x1_coeffs(x1f)` are free
  functions — no mesh, no block, no `MeshBlock`. They return `(cm, cp)` torch
  tensors you can inspect or diff in Python.
* `Reconstruct.set_x1_faces(x1f)` installs them on both interpolators, and they
  are readable as the registered buffers `interp1.cm_x1`, `interp1.cp_x1`,
  `interp2.cm_x1`, `interp2.cp_x1` — through a `Hydro` that is
  `recon1.interp1.cm_x1`.
* What is **not** in stage A: the kernels that consume the rows. `left()` and
  `right()` still run the shipped uniform path, so installing rows does not yet
  change a reconstruction. The `_nu` kernels, their dispatch and the CUDA side
  are stage B.

There is no typed API to hand in a row table you built yourself
(`set_x1_faces` takes faces, not coefficients), and I did not add one: the
coefficients are a function of the faces, and a second entry point would be a
second definition of the same thing. But the buffer **is** the live tensor, so
rows written from Python do stick — `recon.buffer("interp1.cm_x1")[3, 45:] =
torch.tensor([.25, .5, .25])` reads back as `[0.25, 0.5, 0.25]` — which is
enough to drive a harness today. If Xinyue wants a supported entry point rather
than an in-place write, say so and I will add `set_x1_rows` next to it.

**Which interpolator is which.** For a `weno5` deck, `interp1` is `Weno5Interp`
and `interp2` is `Center5Interp` — `src/recon/interpolation.cpp:29-36` branches
on the last character of the module name. So on `recon1` the 48-column table
lives on `interp1` (density and passive scalars,
`src/recon/reconstruct.cpp:139,157`) and the 5-column cp5 table on `interp2`
(the velocities, `:147`). Both are filled by one `set_x1_faces` call.

### The snippet I actually ran

```python
import torch, sys
sys.path.insert(0, "<repo>/python")
import snapy

# 1. the free builder: no mesh, no block
nx1, nghost = 32, 3
nc1 = nx1 + 2 * nghost
q = 1.03
h = torch.cumsum(torch.tensor([0.0] + [q**i for i in range(nc1)],
                              dtype=torch.float64), 0)
x1f = h * 120.0                                   # nc1 + 1 = 39 faces

cm, cp   = snapy.weno5_x1_coeffs(x1f)             # (34, 48) each
cm5, cp5 = snapy.cp5_x1_coeffs(x1f)               # (34, 5)  each

# 2. through a Reconstruct module
opts = snapy.ReconstructOptions()
opts.interp(snapy.InterpOptions("weno5"))
recon = snapy.Reconstruct(opts)
recon.set_x1_faces(x1f)
assert torch.equal(recon.buffer("interp1.cm_x1"), cm)
assert torch.equal(recon.buffer("interp2.cp_x1"), cp5)

# 3. the dotted name from a Hydro
mesh   = snapy.Mesh(snapy.MeshOptions.from_yaml(deck))   # cartesian, weno5
block  = mesh.blocks[0]
phydro = block.module("hydro")
phydro.module("recon1").set_x1_faces(block.buffer("coord.x1f"))
c = phydro.buffer("recon1.interp1.cm_x1")
```

Output:

```
weno5 cm (34, 48) torch.float64  cp (34, 48)
cp5   cm (34, 5)   cp (34, 5)
row 0 linear weights d = [0.3179458049678741, 0.590659866631633, 0.09139432840049298]
row 0 candidate row sums = [1.0, 1.0000000000000002, 0.9999999999999993]
recon.buffer('interp1.cm_x1') -> (34, 48)
recon.buffer('interp2.cm_x1') -> (34, 5)
buffers match the free builder exactly
after in-place write, row 3 d = [0.25, 0.5, 0.25]
block x1f numel = 39 (nc1+1 = 39 )
before: recon1.interp1.cm_x1 -> (0, 48)
after : recon1.interp1.cm_x1 -> (34, 48) torch.float64 cpu
        recon1.interp2.cm_x1 -> (34, 5)
row 0 d = [0.3, 0.6, 0.1]
```

The last line is the uniform deck's own face vector coming back as the shipped
`0.3 / 0.6 / 0.1`, and `(0, 48)` before the call is the empty registered
buffer.

---

## 5. The `extra` for `global_x1f(extra)` — owed to Jiheng

**`extra = max(0, 3 - nghost)`, which is `0` for every shipped deck.**
Equivalently: the WENO5/cp5 stencils close on global faces `-3 .. gnx+3`, i.e.
they need exactly **three** ghost faces beyond each interior end and no more.

Read off the real indices, not assumed:

* output element `m` is the cell window `m .. m+4`
  (`src/recon/interpolation.hpp:64-65`, `stencils() == 5` at
  `src/recon/interpolation.hpp:245`), bounded by faces `m .. m+5`;
* `left()` targets face `m+2`, `right()` targets face `m+3`
  (`src/recon/reconstruct.cpp:54-55`, `il = 3` from `:86-87`);
* so a face `f` is produced twice: as `right()` of `m = f-3`, whose window
  needs faces `f-3 .. f+2`, and as `left()` of `m = f-2`, needing
  `f-2 .. f+3`. Union: faces `f-3 .. f+3`;
* the first interior face is `f = 0` and the last is `f = gnx` in global
  interior numbering, giving `-3 .. gnx+3`.

Today's array already spans global faces `-nghost .. gnx+nghost`
(`src/coord/coordinate.cpp:320-324`), so nothing extra is required as long as
`nghost >= 3`. Every shipped deck sets `nghost: 3`
(`examples/bryan.yaml:28`, `examples/earth_crm.yaml:28`,
`examples/jupiter_crm.yaml:70`, …), so **`extra = 0` in practice**.

Two cautions for piece 1:

1. The default is `nghost = 1` (`src/coord/coordinate.hpp:116`), and I found
   **no** `TORCH_CHECK` anywhere in `src/` that ties `nghost` to the
   reconstruction stencil (the only `nghost` guards are the diffusion ones at
   `src/forcing/diffusion.cpp:382-398`, which ask for `>= 2`). So if
   `global_x1f` is to be safe by construction rather than by deck convention,
   build it with at least three faces per end regardless of `nghost`.
2. This number is **only** the stencil requirement. The **one** extra face per
   end that the note mentions at line 69 is a different thing — it is for
   `dx1v[0]` on the stretched metric fill (`src/coord/cartesian.cpp:28,58-60`),
   which is piece 1's own item. The two add: a global face vector of half-width
   `max(nghost, 3) + 1` satisfies both.

This is derived, not asserted, by `X1Coeffs.ghost_faces_needed_by_the_stencils`,
which reproduces `reconstruct.cpp`'s `il`/`iu`, finds which output element
targets the first and the last interior face, and reports the span. Its output
at `nghost = 3, 4` and `nx1 = 16, 64`:

```
nghost = 3 nx1 = 16: stencils for the first and last interior face span global
  faces -3 .. 19 (interior is 0 .. 16), so they need 3 ghost faces below and 3
  above; extra = max(0, 3 - nghost) = 0
nghost = 4 nx1 = 64: ... span global faces -3 .. 67 (interior is 0 .. 64) ...
  extra = max(0, 3 - nghost) = 0
```

---

## 6. The seam for piece 1

Marked in one place, `src/recon/reconstruct.cpp` immediately above
`ReconstructImpl::set_x1_faces`. When `CoordinateOptions::global_x1f(extra)`
is merged, the call site is `HydroImpl`'s setup right after `recon1` is
created (`src/hydro/hydro.cpp:36`):

```cpp
if (pcoord->options->x1_stretched())
  precon1->set_x1_faces(pcoord->x1f);
```

`pcoord->x1f` is already this block's slice of the global face vector
(`src/coord/coordinate.cpp:300-301,324`), so no extra communication and no
second face array is needed. Piece 1 is unmerged, so nothing calls
`set_x1_faces` today and the tests pass faces in directly. Piece 1 is not
reimplemented or vendored anywhere on this branch.

`REF4` is untouched: `SNAP_WB_REF4`, `src/hydro/wb_ref4.cpp:83-95` and
its gates at `src/hydro/hydro.cpp:581` and `src/hydro/balance_column.cpp:73`
are exactly as shipped. The stretched reference path and the `TORCH_CHECK`
refusal list are Xinyue's piece 3 and are not on this branch.

---

## 7. What stage A deliberately does not contain

`left()` and `right()` run the shipped uniform kernels, unchanged. The `_nu`
kernels (`interp_impl_nu.h/.cuh`), their dispatch
(`recon_dispatch_nu.{hpp,cpp,cu}`), the CUDA build and the G3 bitwise deck are
stage B. A note for that job, not solved here: `call_weno5_cpu`'s
`iter.for_each` lambda (`src/recon/recon_dispatch.cpp:114-123`) does not track
the x1 index, and `for_each` may start a sub-range mid-line, so the loop
counter is **not** reliably the face index.
