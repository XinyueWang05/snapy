# V3 Audit 1 ("one place"): homes in layout, coord, output, eos, recon, riemann, diagnostics, scalar, utils, math

Auditor: XinyueWang05 (Linfeng's former half, reassigned per Xi, ts 1795366481).
This document lists findings only. It changes no source, opens no PR and submits no review.

- Repository `UCzhangxi/snapy` at `v3/wave3` = `d3178f207b60663ff966b7c7ef20025821ef52fa`.
  `git ls-remote` returned that sha at 2026-11-22T16:56:38Z, so the branch had not moved. The command and its output are in the appendix.
- Every `file:line` below is at `d3178f20`.
- Scope: quantities, formulas, constants, option defaults and helpers whose single home is (or should be) in `src/{layout,coord,output,eos,recon,riemann,diagnostics,scalar,utils,math}`.
  I searched all of `src/`, and `python/` where the bindings could duplicate defaults.
- Jiheng owns rows whose home is in `hydro, mesh, implicit, forcing, sedimentation, bc, chemistry, input`.
  When such a row came up I list it under "Handed to Jiheng" and do not count it.
  When a copy in one of my rows sits in one of his directories, the row is marked **crosses Jiheng** so the two halves do not count it twice.
- In-flight work is cited and not counted:
  - **C10** (Linfeng): the recon `shock` default `src/recon/reconstruct.hpp:120` and the shock-path floors.
  - **B1** (Tianhao): register boundary vs `BudgetConsumer::boundary`.
- Method: every row was read by hand at every site listed. Nothing was built or run.
  - The "expected result" column is a static prediction. Where a deck or test is expected to move, the reason is stated and it is a hypothesis until someone runs it.
  - "Net lines" is my estimate of the source delta (added minus removed). It is not measured from a patch.

## Summary

| | count |
|---|---|
| Rows (counted) | **35** |
| … marked *crosses Jiheng* | 9 (L1, L2, L4, L7, G4, G8, E1, R2, R4) |
| … predicted bitwise on every in-tree deck and test | 29, of which 3 carry a behaviour note outside the tests (G8: only with `SNAP_X1_CENTROID_EXACT` plus a scalar; E1: aneos with roe or outflow bc; C2: the scalar reader accepts a key it ignores) |
| … predicted to move some deck, test or output (reason in the row; all hypotheses, not run) | 6 (G2 spherical-polar ULP, G4 gnomonic wall ULP, G5 one cubed-sphere side ULP, E3 output `theta_v` only, E4 budget/cycle-line `ke` only, O3 `NumFilesInSet` on stretched x1 slices) |
| Cited, not counted (C10 / B1) | 2 |
| Handed to Jiheng, not counted | 5 |
| Estimated net lines, all rows | ≈ −288 in `src/` (plus ≈ −20 in `tests/test_options_table.cpp` if L8/G1 land) |

Per home: layout 8, coord 8, eos 7, riemann 5, output 4, recon 2, scalar 1. Diagnostics has no counted row of its own: its KE copy is in E4 and its boundary copy is B1. utils' only row is R4, filed under riemann. math has no finding.

## Table

Columns: what | every site file:line (at d3178f20) | which stays | what the others become | expected result | net lines.

### layout

| # | what | every site | which stays | what the others become | expected result | net lines |
|---|---|---|---|---|---|---|
| L1 | Block-count axis map `nb1 = pz`, `nb2 = px`, `nb3 = py` (and its inverse in the reader), re-derived at each use. **crosses Jiheng** | `src/layout/layout.cpp:270-272` (reader: nb3→py, nb2→px, nb1→pz); `src/coord/coordinate.cpp:188-190`; `src/coord/coordinate.cpp:348-350`; `src/output/writer_core.cpp:264-266`; `src/output/netcdf.cpp:164-168`; `src/coord/gnomonic_equiangle.cpp:234` (`px` as "x2 blocks"); `src/forcing/diffusion.cpp:466` (Jiheng); `python/csrc/pymesh.cpp:155-156` | A new `LayoutOptionsImpl::nb(int axis)` (axis 1..3), next to `px/py/pz` in `src/layout/layout.hpp` | Callers use `nb(1..3)`. The reader keeps its key mapping in one place: the table row in `src/options/layout.yaml` (see L8) | Bitwise (integer renaming only) | ≈ −8 |
| L2 | Process-grid size `px*py*pz` (or `px*py` for slab) recomputed. **crosses Jiheng** | `src/layout/layout.hpp:217`; `src/layout/cubed_layout.cpp:28`, `:35`; `src/layout/slab_layout.cpp:30`, `:37` (pz==1 asserted at `:12`); `src/mesh/meshblock_restart.cpp:178` (Jiheng); `src/mesh/stage_executor.cpp:88` (Jiheng) | `LayoutImpl::nblocks()` in `layout.hpp` | Every site reads `nblocks()` | Bitwise | ≈ −1 |
| L3 | Rank arithmetic copied: the periodic wrap / reject block (5 copies) and a hand-written `linear_index3` | wrap: `src/layout/slab_layout.cpp:49-56`, `:58-65`; `src/layout/cubed_layout.cpp:50-57`, `:59-66`, `:68-75`. Index: `src/layout/layout.hpp:261` (`rz*(px*py)+ry*px+rx`) duplicates `src/layout/connectivity.hpp:156` (`linear_index3`) | `connectivity.hpp`: `linear_index3`, plus a new `wrap_or_reject(n, p, periodic)` | Five 8-line blocks become one call each. `layout.hpp:261` calls `linear_index3` | Bitwise (same integer formula) | ≈ −32 |
| L4 | The layout-type test `type() == "cubed-sphere"` and the panel of a rank, `std::get<2>(loc_of(r))`, spelled out at each use. **crosses Jiheng** | type test: `src/coord/coordinate.cpp:186`, `:346`; `src/coord/gnomonic_equiangle.cpp:79`; `src/layout/exchange.cpp:47`; `src/output/netcdf.cpp:105`; `src/output/writer_core.cpp:269`; `src/scalar/scalar.cpp:125`; `src/forcing/diffusion.cpp:436`; `src/hydro/hydro_forward.cpp:706`, `:718`, `:725`, `:863`; `src/mesh/meshblock.cpp:121`; `src/mesh/meshblock_exchange.cpp:40`, `:119`. Panel of rank: `src/coord/gnomonic_equiangle.cpp:80`; `src/eos/equation_of_state.cpp:415`; `src/forcing/coriolis.cpp:70`; `src/layout/cubed_sphere_layout.cpp:187`, `:586`, `:637`, `:764`, `:792`; `src/layout/exchange.cpp:48`, `:205`; `src/layout/layout.cpp:221`; `src/layout/seam_register.cpp:53`; `src/output/load_hydro_output_data.cpp:144` | `LayoutImpl::is_cubed_sphere()` and `LayoutImpl::panel_of(int rank)` in `layout.hpp` | Every site calls them. The `"cubed-sphere"` literal stays only in `LayoutImpl::create` (`layout.cpp`) and `CubedSphereLayoutImpl`'s constructor | Bitwise | ≈ −4 |
| L5 | "Reuse the buffer if it matches, else allocate", repeated at each exchange buffer. Two more copies sit outside layout | `src/layout/layout.cpp:540-542`, `:545-548`; `src/layout/cubed_sphere_layout.cpp:604-606`, `:607-609`, `:672-674`, `:675-677`, `:678-680`; also `src/scalar/scalar.cpp:83-88` (`put`) and `src/riemann/riemann.cpp:234-238` (set_data re-allocation of `elr/clr/glr`) | `buffer_matches` (`src/layout/layout.cpp:190`, declared in `layout.hpp:173`), extended to an `ensure_like(buf, src)` that allocates when it does not match | Seven 3-line blocks become `ensure_like(...)`. `scalar.cpp put` becomes `ensure_like` + `copy_` | Bitwise (allocation only; values copied as before) | ≈ −17 |
| L6 | The cubed-sphere velocity transform switch (`kConserved` with metric cosine, `kPrimitive` without, `kScalar` skip) appears in both serialize and deserialize | `src/layout/cubed_sphere_layout.cpp:686-701` (serialize, to_global) and `:823-839` (deserialize, from global) | A private `CubedSphereLayoutImpl::_transform_velocity_(opts, bid, to_global, alpha, beta, cosine, vel)` | The two switches become one call each | Bitwise (same calls, same order) | ≈ −12 |
| L7 | x1 exchange message tags are scattered literals with no registry. Today they do not collide (I checked the ranges by hand: `0x7740+4k` with k∈{0,1} at hydro_forward.cpp:236/257). **crosses Jiheng** | `src/recon/x1_reference.cpp:58-59` (0x7715, 0x7717); `src/hydro/hydro_forward.cpp:257` (0x7740+4k), `:369` (0x772a), `:477` (0x7724), `:596` (0x7720), `:655` (0x7722), `:699-700` (0x7730, 0x7734), `:826` (0x7750), `:1106` (0x7726); `src/hydro/sweep.hpp:25` (comment copy of 0x7730/0x7734); `src/forcing/diffusion_cubed_sphere.cpp:102` (0x7760) | A tag enum in `src/layout/exchange.hpp`, which already documents the tag / tag+1 convention at `:51`, `:64`, `:82` | Each site names its enum entry. The `sweep.hpp` comment points at the enum | Bitwise (same integers) | ≈ +7 |
| L8 | Layout option defaults live in three places. The generated reader for `src/options/layout.yaml` (`DistributeRows`) is used only by a test. The production reader still hand-copies the literals | ADD_ARG: `src/layout/layout.hpp:102`, `:105`, `:108`, `:111`, `:123`. Hand reader: `src/layout/layout.cpp:269-273` (`"slab"`, `1`, `1`, `1`, `"cpu"`). Table: `src/options/layout.yaml:17-23`. Only consumer of the table: `tests/test_options_table.cpp:282-292` | `src/options/layout.yaml` (through `DistributeRows::read`), as `output.yaml` / `hydro.yaml` already are (`src/output/output_type.cpp:44`, `src/hydro/hydro_options.cpp:17`) | `LayoutOptionsImpl::from_yaml` maps `DistributeRows` onto the options. The literals at `layout.cpp:269-273` go. The test's reader-vs-table cross-check becomes trivially true and can shrink | Bitwise on every in-tree deck (the G-3 test already asserts that table and reader agree on all of them, `tests/test_options_table.cpp:282-292`) | ≈ −4 (src), ≈ −10 (test) |

### coord

| # | what | every site | which stays | what the others become | expected result | net lines |
|---|---|---|---|---|---|---|
| G1 | Geometry option defaults live in three places. The table reader `GeometryRows` (from `src/options/coord.yaml`) is used only by a test. The hand reader also keeps its own local default block | ADD_ARG: `src/coord/coordinate.hpp:111-131`. Hand reader: `src/coord/coordinate.cpp:71` (`"cartesian"`), `:73` (local `x1min = 0 … x3max = 1`), `:79`, `:84`, `:90`, `:93`, `:98`, `:104` (`0.0`/`1.0`), `:213` (`nghost` 1), `:214` (`interp_order` 2). Table: `src/options/coord.yaml:16-29`. Only consumer: `tests/test_options_table.cpp:253-276` | `src/options/coord.yaml` via `GeometryRows::read` (it already handles the `_pi` aliases and x1f/x1rat checks, `coord.yaml:20-23`, `:31-39`) | `CoordinateOptionsImpl::from_yaml` keeps only the grid rules (one cell per axis without `cells`, the x1rat faces, the repartition) and reads values from `GeometryRows`. All 10 literal sites and the local block go | Bitwise on in-tree decks (the G-3 test asserts agreement on every deck that loads) | ≈ −25 (src), ≈ −10 (test) |
| G2 | Spherical shell measures written out repeatedly: radial volume `(rp³−rm³)/3`, radial area `½(rp²−rm²)`, polar `|cos θm − cos θp|`, and the `rm/rp/h/rbar` prologue. Gnomonic computes the radial volume in a different (cancellation-free) form | volume: `src/coord/spherical_polar.cpp:80`, `:154-158`; `src/coord/gnomonic_equiangle.cpp:288-291` (`dr(rp²+rp rm+rm²)/3`). Area: `spherical_polar.cpp:81` (`pow(2)`), `:139-140`, `:147-148` (`square()`). Polar: `spherical_polar.cpp:87`, `:132-133`, `:159-160`. Prologue: `src/coord/coordinate.cpp:620-623`, `:641-644` (also the degenerate mask `rbar > 0.5*h` at `:628`, `:659`) | New static helpers in `CoordinateImpl` next to `radial_face_moment2_` (`coordinate.hpp`), with the cancellation-free volume form | All sites call the helpers | **Not bitwise for spherical-polar.** The single volume form moves `cell_volume()` / `coord_src*_i` at round-off, so spherical-polar decks and tests move at ULP level (hypothesis, not run). Gnomonic stays bitwise. If bitwise is required, keep two named helpers (one per form) and the row still removes 7 copies. `pow(2)` vs `square()`: bitwise is likely but untested (hypothesis) | ≈ −20 |
| G3 | The uniform x2/x3 axis metric block and the zero cosine buffers are copy-pasted between coordinate types | x2/x3 metrics: `src/coord/cartesian.cpp:28-40` = `src/coord/gnomonic_equiangle.cpp:58-70`. Zero cosines: `src/coord/cartesian.cpp:11-20` = `src/coord/spherical_polar.cpp:50-57` | A base helper `CoordinateImpl::uniform_axis_metrics_(axis)` beside `x1_metrics_()` (`coordinate.cpp:436`), plus `zero_cosines_()` | The three types call them | Bitwise | ≈ −20 |
| G4 | The sine of the panel angle is recomputed as `sqrt(1−c²)` while gnomonic stores the exact sine. **crosses Jiheng** | `src/coord/coordinate.cpp:770-771` (`boundary_velocity_`); `src/forcing/diffusion_cubed_sphere.cpp:234` (Jiheng). Stored: `src/coord/gnomonic_equiangle.cpp:99`, `:102`, `:105` | `sine_cell_kj` / `sine_face{2,3}_kj` moved to the base (ones for orthogonal types) | Both sites read the stored sine | Cartesian / spherical-polar bitwise (c = 0 gives s = 1 exactly). **Gnomonic not bitwise:** the boundary velocity on x2/x3 physical boundaries (`bc_func.cpp:170-171`, `:220`) and the cubed-sphere diffusion basis change at round-off. A regional gnomonic deck with a physical x2/x3 wall would move (hypothesis, not run) | ≈ +4 |
| G5 | Gnomonic metric factors `C = √(1+x²)`, `D = √(1+y²)`, `δ = √(1+x²+y²)`, three copies | `src/coord/cubed_sphere_utils.cpp:199-201`, `:226-228`; `src/coord/gnomonic_equiangle.cpp:93-96`, `:99`, `:102`, `:105` (δ inline) | One `cs_metric(x, y)` helper in `cubed_sphere_utils.hpp` | All three call it | C and D bitwise. δ is written `x*x+y*y+1` in utils and `1+x*x+y*y` in gnomonic, so unifying the order moves one side at ULP. That changes either the cubed-sphere velocity transforms or the stored `sine_*_kj`, and cubed-sphere decks would move at round-off (hypothesis). Keeping utils' order leaves the exchanges bitwise and moves only the stored sine | ≈ 0 |
| G6 | The length guard `clamp_min(1.e-30)` is a literal repeated in `derivative` and `curl` | `src/coord/coordinate.cpp:37`, `:50`, `:57`, `:731`, `:732`, `:733`, `:734`, `:735`, `:736` | One `constexpr double kTinyLength` in `coordinate.cpp` | The nine sites use it | Bitwise | ≈ +1 |
| G7 | `CoordinateImpl::print` copy-pastes the same loop six times | `src/coord/coordinate.cpp:471-518` (x1f, x1v, x2f, x2v, x3f, x3v) | A local lambda `put(name, tensor)` | Six loops become six calls | Bitwise (same text) | ≈ −34 |
| G8 | Which x1 faces a reconstruction gets (`x1_row_faces`, the `plain` spherical exception under `SNAP_X1_CENTROID_EXACT`, the stretched fallback) is decided in three places. The scalar copy lacks the `plain` exception. **crosses Jiheng** | `src/hydro/hydro.cpp:51-58` (Jiheng); `src/recon/x1_reference.cpp:190-196`; `src/scalar/scalar.cpp:45-50` (no `plain` branch) | `CoordinateImpl::x1_row_faces` (`src/coord/coordinate.cpp:521`) absorbs the `plain` rule | The three sites call it | Bitwise when `SNAP_X1_CENTROID_EXACT` is unset. Seven ctest entries set it (`tests/CMakeLists.txt:54`, `:125`, `:133`, `:141` CUDA only, `:160`, and `:171` for two tests); `tests/test_radial_order.py:11` sets it too, and none of their sources mention `scalar` (grep output 039), so I expect every test to stay bitwise (hypothesis). With it set, on spherical-polar weno5/cp5 the scalar's x1 recon changes (the hydro and the WB reference are unchanged). It is a behaviour question for the scalar's owner whether its current divergence is intended | ≈ −4 |

### eos

| # | what | every site | which stays | what the others become | expected result | net lines |
|---|---|---|---|---|---|---|
| E1 | The sound speed + adiabatic index pair, including the aneos rule ("W->L then WL->A; otherwise W->A then WA->L"), is spelled out by each caller. **crosses Jiheng** | `src/riemann/riemann.cpp:251-257` (hllc), `:263-268` (lmars), `:284-289` (roe: no aneos branch); `src/hydro/hydro.cpp:338-343` (Jiheng); `src/hydro/hydro.cpp:472-477` (Jiheng; `gamma_for(state)` passes `w`, not `state`, to `W->L` under aneos, so the argument is ignored there: a latent defect, untested); `src/bc/bc_func.cpp:156-157` (Jiheng; no aneos branch) | A new `EquationOfStateImpl::gamma_sound_speed(w)` returning `{gamma, c}` with the aneos rule inside (overridden by aneos) | Every caller asks the EOS. riemann's roe and bc gain the aneos branch through it | Bitwise for every non-aneos deck (same keys, same order). The aneos + roe / outflow-bc combinations change, but they are probably unsupported today (hypothesis; I did not check whether any deck combines them) | ≈ −8 |
| E2 | `"WA->L"` = `sqrt(gamma p / rho)` written identically in two EOS types | `src/eos/ideal_gas.cpp:49-54`; `src/eos/ideal_moist.cpp:195-200` | The base `EquationOfStateImpl::compute` (or `compute_common_`) | The two branches go | Bitwise | ≈ −7 |
| E3 | The dry gas constant `Rd` is recomputed at each use, and there are two definitions in play: `Rgas/weight` and `Rgas*inv_mu[0]` (weight = `1/inv_mu[0]` via `species_weight(0)`, `moist_base.cpp:21`, `:33`) | `Rgas/weight`: `src/eos/ideal_gas.cpp:21`, `:40`; `src/eos/ideal_moist.cpp:256`, `:265`, `:276`, `:298-299`. `Rgas*inv_mu[0]`: `src/eos/ideal_moist.cpp:45`. Output: `src/output/load_diag_output_data.cpp:100` (`Rgas/mu[0]` of a fresh ThermoX) | `EquationOfStateImpl::gas_constant_dry()` = `Rgas / options->weight()` | All six eos sites call it. `ideal_moist.cpp:45` keeps `inv_mu` (its value is a ratio of two `inv_mu` terms; leave it). The output site calls the EOS | EOS sites bitwise. The output `theta_v` may move at ULP if `mu[0]` and `1/(1/mu[0])` differ in the last bit (hypothesis); it affects output only | ≈ −2 |
| E4 | Kinetic energy from conserved variables, `½ m·raise(m) / ρ_total`, with ρ_total = dry + Σ species, written out three more times | Home: `src/eos/moist_base.cpp:127-136` (`_cons2ke`, exposed as `compute("U->K")` at `:63-64` for the moist types only). Copies: `src/eos/equation_of_state.cpp:318-325` (limiter); `src/diagnostics/budget.cpp:316-321` (cycle line; ρ summed sequentially, `:317-318`). Read and **excluded**: `src/eos/ideal_moist.cpp:126-131` uses the anomaly split density by design | `compute("U->K")` / `_cons2ke`, given a base implementation in `EquationOfStateImpl` so every type has it | The limiter and the budget call it | Hydro bitwise (the limiter uses the same `dry_total + species_total` order). Budget `ke=` and its cycle line: ρ becomes `ρ_d + (y₀+y₁+…)` instead of `((ρ_d+y₀)+y₁)…`, a ULP change for decks with ≥2 species. It affects diagnostics only and may not show at the print precision (hypothesis) | ≈ −8 |
| E5 | Kinetic energy `½ v·m` from primitive and conserved variables, repeated in every conversion | `src/eos/aneos.cpp:55`, `:75`; `src/eos/ideal_gas.cpp:70`; `src/eos/moist_base.cpp:93`, `:123`; `src/eos/ideal_moist.cpp:285-287` and `src/eos/moist_mixture.cpp:122-124` (lower `v`, then dot); kernel `src/eos/ideal_gas_impl.h:29-30` | A base `EquationOfStateImpl::kinetic_energy_(prim, mom)` (torch). The kernel keeps its scalar form | The torch sites call it | Bitwise (same expression) | ≈ −2 |
| E6 | Species counts recomputed from the thermo card instead of `thermo_species_rows` / `nspecies()`. Note that `nvapor` means "vapors without dry" in ideal_moist and "vapors with dry" in moist_mixture | `src/eos/equation_of_state.cpp:310-317`; `src/eos/ideal_moist.cpp:262`, `:273-274`, `:320-321`; `src/eos/moist_mixture.cpp:109-110`, `:150` (`nvapor` includes dry); `src/riemann/riemann.cpp:277`; `src/output/load_diag_output_data.cpp:61-62` | `thermo_species_rows` (`src/eos/moist_base.hpp:10-14`) and a sibling `thermo_vapor_rows` | All sites call them; `moist_mixture`'s local becomes `ngas` | Bitwise (integers), assuming `species().size() − 1` equals vapor+cloud−1 for the output site (hypothesis; true for every card I read) | ≈ −6 |
| E7 | The EOS hand reader repeats five ADD_ARG defaults. The floors already read `op->…`, so the file mixes both styles | ADD_ARG `src/eos/equation_of_state.hpp:46-48`, `:56-57`. Reader `src/eos/equation_of_state.cpp:96` (`"moist-mixture"`), `:101` (1.4), `:102` (29.e-3), `:115` (false), `:121` (`""`) | `equation_of_state.hpp` ADD_ARG | The reader uses `op->type()` etc. as `:104-111` already do (or a YAML table like hydro's) | Bitwise | ≈ 0 |

### riemann (and utils)

| # | what | every site | which stays | what the others become | expected result | net lines |
|---|---|---|---|---|---|---|
| R1 | `TINY_NUMBER = 1.0e-10` defined twice | `src/riemann/riemann_kernels.h:41` (hllc, constexpr), `:279` (roe, auto) | One namespace-scope `template <class T> constexpr T kTiny` in `riemann_kernels.h` | Both use it | Bitwise | ≈ −1 |
| R2 | Face-local velocity permutation (`ivx = IPR−dim`, `ivy`, `ivz` cyclic), plus the shallow-roe `dir_yz` map twice. **crosses Jiheng** (implicit has its own `dir`-based cyclic map, a different convention, listed for the other half) | `src/riemann/riemann_kernels.h:43-45`, `:192-194`, `:281-283`, `:568`; shallow: `:480-488`, `:564`. Related, different convention: `src/implicit/flux_decomposition_impl.h:82-83`, `:112-113`; `src/implicit/forward_sweep_impl.h:71-72`; `src/implicit/vic_*_impl.h` | A `DISPATCH_MACRO face_axes(dim, &ivx, &ivy, &ivz)` and `shallow_axes(dim, dir_yz, …)` in `riemann_kernels.h` | Four plus two sites call them | Bitwise (integers) | ≈ −8 |
| R3 | Dry fraction `1 − Σ y_n` loops | kernels: `src/riemann/riemann_kernels.h:119-124` (hllc), `:221-224` and `:237-239` (lmars, per side), `:300-305` (roe), `:571-575` (`riemann_central`); torch: `src/hydro/hydro_forward.cpp:243-246` (Jiheng); species_total form: `src/eos/moist_base.cpp:81`, `src/eos/ideal_moist.cpp:247` | A device helper beside `ideal_moist_feps` in `src/eos/ideal_moist_impl.h` that keeps the sequential order, plus its torch twin in `src/eos/species_sum.hpp` | Kernel loops call the device helper | Kernels bitwise (same order). Do **not** merge with the `1 − species_total(y)` form: that associates differently and would move decks with ≥2 species at ULP. Move `hydro_forward.cpp:243-246` onto the sequential twin only | ≈ −8 |
| R4 | Kernel macro block `SQR/WL/WR/FLX` defined three times in one header, plus two more square macros. **crosses Jiheng** | `src/riemann/riemann_kernels.h:22-25`, `:181-184`, `:258-263` (with `#undef`s at `:176-179`, `:255-258`, `:465-468`); `src/recon/recon_point.h:12` (`SNAP_SQR`); `src/implicit/flux_decomposition_impl.h:12` (Jiheng) | One `template <class T> DISPATCH_MACRO constexpr T sqr(T)` in a `src/utils` header, with the WL/WR/FLX macros defined once at the top of `riemann_kernels.h` and undefined once at the bottom | The redefinition blocks go. `SNAP_SQR` and the implicit `SQR` use `sqr` | Bitwise (same expression) | ≈ −16 |
| R5 | `RiemannSolverOptionsImpl::from_yaml(filename, section)` tests `config[section]` but reads `config["dynamics"]` (the literal is repeated three times). The `{"type","dir"}` key check is copied in scalar | `src/riemann/riemann.cpp:53-63` (tests `config[section]` at `:58-59`, reads `config["dynamics"]` at `:60-62`); `src/scalar/scalar_options.cpp:46-47` | `RiemannSolverOptionsImpl::from_yaml(node)` (`riemann.cpp:65-74`) owns the key check; the filename overload uses `section` | The scalar drops its copy. The `"dynamics"` literal goes | Bitwise: the only caller passes `"dynamics"` (`src/hydro/hydro_options.cpp:35`) | ≈ −2 |

### recon

| # | what | every site | which stays | what the others become | expected result | net lines |
|---|---|---|---|---|---|---|
| C1 | WENO weight epsilon `T(1e-6)` five times, and the `vscale`/`phi` prologue twice | `src/recon/recon_point.h:88`, `:89`, `:135`, `:136`, `:137`; prologue `:70-80` and `:110-121` | One `constexpr` epsilon and a `weno_scale<N>(src, stride, scale)` helper in `recon_point.h` | All sites use them | Bitwise | ≈ −2 |
| C2 | Interp option defaults (`type` "dc", `scale` false) repeated, and the scalar's `reconstruct:` reader is a second copy of `ReconstructOptionsImpl::from_yaml`. **Touches C10:** the `shock` literal sits in both copies (`src/recon/reconstruct.cpp:205`, `src/scalar/scalar_options.cpp:37`, vs ADD_ARG `src/recon/reconstruct.hpp:120`). C10 owns the `shock` default; only the `type`/`scale` part is counted here | ADD_ARG `src/recon/reconstruct.hpp:39-40`; readers `src/recon/reconstruct.cpp:207-208`, `src/scalar/scalar_options.cpp:38` | `reconstruct.hpp` ADD_ARG, and one node-level `ReconstructOptionsImpl::from_yaml(node)` | The readers use `op->interp()->type()` etc. The scalar calls the shared node reader (after C10 lands, so the `shock` literal is not touched twice) | Bitwise on decks. Behaviour: `scalar/reconstruct` would also accept `scale:` (the scalar forces `scale = true` anyway at `src/scalar/scalar.cpp:43`), so it only accepts a key it ignores today | ≈ −3 |

### scalar

| # | what | every site | which stays | what the others become | expected result | net lines |
|---|---|---|---|---|---|---|
| S1 | Scalar reader repeats ADD_ARG defaults | ADD_ARG `src/scalar/scalar.hpp:60` (nvar 0), `:72` (upper_bound −1.); reader `src/scalar/scalar_options.cpp:26-27` | `scalar.hpp` ADD_ARG | The reader uses `op->nvar()`, `op->upper_bound()` | Bitwise | ≈ 0 |

### output

| # | what | every site | which stays | what the others become | expected result | net lines |
|---|---|---|---|---|---|---|
| O1 | Cubed-sphere lon/lat recomputed for output, although gnomonic already registers them as buffers | `src/output/load_hydro_output_data.cpp:142-157`; buffers `src/coord/gnomonic_equiangle.cpp:76-85` | The `longitude`/`latitude` buffers | Output appends `buffer.squeeze(-1)` | Bitwise (same function, same `x3v/x2v` mesh, same face id) | ≈ −8 |
| O2 | 3×2 panel placement `face % 3`, `face / 3` | `src/output/writer_core.cpp:272-273`; `src/output/netcdf.cpp:132`, `:202`; `src/output/pnetcdf.cpp:160` | `output_block_location` (`writer_core.cpp:260-277`) gains `panel_col/panel_row` | The writers read them | Bitwise | ≈ 0 |
| O3 | netcdf recomputes the block count (`pz/px/py`, then ×6 for cubed-sphere) and each block's bounds with a uniform formula. `output_block_location` already returns the counts (with ×3×2), and the coord repartition owns the bounds, including stretched x1 | `src/output/netcdf.cpp:150-169`; vs `src/output/writer_core.cpp:264-276` and `src/coord/coordinate.cpp:353-368` | The coord's block bounds (exposed as a helper from the repartition) and `output_block_location`'s counts | `count_blocks` uses them | Bitwise on uniform grids. On a stretched x1 with `x1_slice`, `NumFilesInSet` changes to the true count. That is a fix, and it would move any test pinning the current value (hypothesis: I found none by reading, and did not run) | ≈ −8 |
| O4 | netcdf and pnetcdf copy the coordinate-attribute block and the per-field axis/ndim table | coordinate attrs `src/output/netcdf.cpp:115-123` ≈ `src/output/pnetcdf.cpp:102-111`; field table `netcdf.cpp:174-192` = `pnetcdf.cpp:127-145` | Helpers in `writer_core` (a `field_dims(kind, ids…)` and the attribute text) | Both writers call them; only the library call differs | Bitwise (same files) | ≈ −30 |

## Cited, not counted

| ref | what | sites (d3178f20) | owner |
|---|---|---|---|
| C10 | recon `shock` default: ADD_ARG `true` vs readers `false`, plus the shock-path floors | `src/recon/reconstruct.hpp:120`; `src/recon/reconstruct.cpp:205`; `src/scalar/scalar_options.cpp:37`; shock path `src/recon/reconstruct.cpp:282-288` (species floor only, no density/pressure floor) | Linfeng (C10). C2 above defers to it |
| B1 | Outward boundary transfer defined twice: register boundary vs `BudgetConsumer::boundary` | `src/diagnostics/budget.cpp:78-155` (with `boundary_face` `:52-62`) | Tianhao (B1, option (a)) |

## Handed to Jiheng (home in his directories; not counted)

These copies were found from my side. Their home is in hydro/mesh/bc, so they are listed for his half:

1. The "no limit" time-step sentinel: `src/scalar/scalar.hpp:114` (1.e9), `src/hydro/hydro.cpp:350` and `:387` (1.e9), `src/mesh/meshblock_advance.cpp:76` (1.e9), `src/mesh/mesh.cpp:344` (1.e99). Home: mesh, which consumes it.
2. The gravity potential `−grav1·x1v` recomputed by diagnostics: `src/diagnostics/budget.cpp:325` vs `src/hydro/gravity_work.hpp:302` and `src/hydro/hydro_forward.cpp:1056`, `:1070`. Home: hydro.
3. Cross-panel L/R face-state exchange (direction-suffixed keys, begin/launch/finalize): `src/scalar/scalar.cpp:125-143` vs `src/hydro/hydro_forward.cpp:702-740`. Home: mesh (`begin_exchange`).
4. The interior part `part({0,0,0}, PartOptions().exterior(false))` with a sum over it, repeated in eos, diagnostics, scalar, hydro and mesh (grep output 024). Home: mesh.
5. `src/options/internal_boundary.yaml` / `InternalBoundaryRows` is likewise test-only while `src/bc/internal_boundary.cpp:32-33` hand-reads `1.e3` and `1.e9` (also ADD_ARG `src/bc/internal_boundary.hpp:50-51`). Same pattern as L8/G1. Home: bc.

## Not checked

- No build, no ctest, no deck run. Every "bitwise" or "moves" call is a static prediction, and the moves are labelled hypotheses.
- Net lines are estimates, not measured from patches.
- `src/output/restart.cpp`, `combine_parts.cpp`, `combine_netcdf.cpp`, `budget_series.cpp` and `src/layout/process_group*.cpp` were skimmed by grep only, not read line by line.
- CUDA `.cu` files were read for their dispatch tables only. The CPU and CUDA dispatch files differ substantively (loop vs launch), and I filed no row on them. I did confirm that `recon_dispatch.cu` passing `scale=false` for cp3/cp5 is harmless, because `recon_point.h:146-150` ignores `scale` for polynomial schemes.
- `tests/` and `examples/` were not searched for copies of formulas (only for consumers of the option tables).
- `python/`: the bindings and `.pyi` stubs carry no duplicated value defaults for my homes (grep outputs 027-030). The `pyeos.cpp:50` keyword `verobse` is a typo, not a one-place finding.
- `src/math` (`ludcmp.h`, `lubksb.h`): its only callers are `src/implicit/forward_sweep_impl.h`. `src/recon/x1_rows.cpp:276`, `:279` use `torch::inverse` on small tensors, which is not a reimplementation. No finding.
- `src/utils`: `TensorCastCache` and `OnHost` have no local re-implementations that I found (grep outputs 032-034). `CoordinateImpl::cached_geometry_` (`coordinate.cpp:564-588`) is a version cache with a different contract (rebuild on input change). Not filed.

## Appendix: commands and outputs

The worktree was created from the existing local checkout (`origin` = `https://github.com/UCzhangxi/snapy.git`) with:

```
git fetch origin v3/wave3          ->  c33a67f9..d3178f20  v3/wave3 -> origin/v3/wave3
git cat-file -t d3178f207b60663ff966b7c7ef20025821ef52fa   ->  commit
git worktree add -b v3/audit1-xinyue ../wt/audit1 d3178f207b60663ff966b7c7ef20025821ef52fa
                                    ->  HEAD is now at d3178f20 recon, riemann, hydro: lane C under its 4,672-line ceiling (V3 C.trim)
gh api user --jq .login             ->  XinyueWang05
gh repo view XinyueWang05/snapy --json name,parent  ->  snapy UCzhangxi   (existing fork used)
```

Reading commands (outputs were the source text itself, read by hand; listed here with what they covered):

```
find src -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.h' \) | awk -F/ '{print $2}' | sort | uniq -c
    -> layout 18, coord 22, output 19, eos 25, recon 13, riemann 5, diagnostics 3, scalar 3, utils 3, math 2 (+ Jiheng's dirs)
wc -l per in-scope dir -> layout 3927, coord 3729, output 3151, eos 2729, recon 2509, riemann 1349, diagnostics 534, scalar 414, utils 342, math 166
sed -n 1,609p src/riemann/riemann_kernels.h            (whole file)
cat src/riemann/riemann.cpp                             (whole file)
cat src/eos/species_sum.hpp src/snap.h src/eos/ideal_moist_impl.h src/eos/ideal_gas_impl.h src/eos/shallow_water_impl.h
sed -n 325,400p / 462,482p src/hydro/hydro.cpp ; sed -n 145,165p src/bc/bc_func.cpp
cat src/eos/equation_of_state.hpp ; sed -n 60,140p / 245,262p / 300,335p src/eos/equation_of_state.cpp
grep -n ADD_ARG (coord, layout, recon, scalar, riemann, output, diagnostics headers)
sed -n 60,225p / 225,520p / 520,794p src/coord/coordinate.cpp      (whole file)
sed -n 195,300p src/layout/layout.hpp ; cat src/layout/slab_layout.cpp src/layout/cubed_layout.cpp
grep -n linear_index src/layout/connectivity.hpp + users
sed -n 40,130p src/layout/exchange.cpp ; sed -n 255,300p / 425,450p / 500,620p src/layout/layout.cpp
sed -n 1,264p src/coord/spherical_polar.cpp (whole) ; sed -n 1,300p src/coord/gnomonic_equiangle.cpp ; cat src/coord/cartesian.cpp
sed -n 1,118p src/layout/cubed_sphere_layout.hpp ; cat src/layout/cubed_sphere_constants.h ; sed -n 350,370p / 570,892p src/layout/cubed_sphere_layout.cpp
sed -n 180,336p src/coord/cubed_sphere_utils.cpp
cat src/output/load_hydro_output_data.cpp src/output/load_diag_output_data.cpp
sed -n 90,264p src/output/netcdf.cpp ; sed -n 60,216p src/output/pnetcdf.cpp ; sed -n 250,290p src/output/writer_core.cpp
sed -n 1,109p src/eos/ideal_gas.cpp ; sed -n 30,350p src/eos/ideal_moist.cpp (whole) ; sed -n 30,179p src/eos/moist_mixture.cpp
sed -n 1,138p src/eos/moist_base.cpp ; sed -n 1,30p src/eos/moist_base.hpp ; sed -n 40,80p src/eos/aneos.cpp ; sed -n 10,56p src/eos/shallow_water.cpp
sed -n 100,201p src/recon/reconstruct.hpp ; cat src/recon/reconstruct.cpp ; sed -n 1,163p src/recon/recon_point.h
cat src/recon/conversion.cpp ; sed -n 1,80p / 185,200p src/recon/x1_reference.cpp
diff src/recon/recon_dispatch.cpp src/recon/recon_dispatch.cu (and the riemann/coord/eos pairs: 156/87/147/87 differing lines); tails of recon_dispatch.{cpp,cu}, riemann_dispatch.{cpp,cu}
cat src/scalar/scalar_options.cpp ; sed -n 50,126p src/scalar/scalar.hpp ; sed -n 1,223p src/scalar/scalar.cpp
sed -n 695,735p / 225,258p src/hydro/hydro_forward.cpp ; sed -n 45,62p src/hydro/hydro.cpp
cat src/diagnostics/consumer.hpp ; sed -n 1,394p src/diagnostics/budget.cpp (whole)
cat src/options/layout.yaml src/options/coord.yaml ; sed -n 1,73p src/options/output.yaml ; grep -n name: src/options/hydro.yaml
sed -n 1,40p / 240,300p tests/test_options_table.cpp
sed -n 1,53p src/utils/tensor_cast_cache.hpp
wc -l src/output/restart.cpp combine_parts.cpp combine_netcdf.cpp budget_series.cpp src/layout/process_group.cpp -> 408 178 71 232 297
```

Logged search commands, with their output in full or trimmed (the counts are shown when trimmed):

**000**

```
$ git rev-parse HEAD; git ls-remote https://github.com/UCzhangxi/snapy refs/heads/v3/wave3; date -u +%FT%TZ
d3178f207b60663ff966b7c7ef20025821ef52fa
d3178f207b60663ff966b7c7ef20025821ef52fa	refs/heads/v3/wave3
2026-11-22T16:56:38Z
```

**001**

```
$ grep -rnoE '\b[0-9]+\.[0-9]*e-?[0-9]+\b|\b1e-?[0-9]+\b' src --include=*.cpp --include=*.hpp --include=*.h --include=*.cu --include=*.cuh | awk -F: '{print $NF}' | sort | uniq -c | sort -rn | head -30
     10 1.e-30
      8 1.e9
      5 1e-6
      3 1.e-300
      3 1.e-10
      2 29.e-3
      2 1e-7
      2 1.e-6
      2 1e-300
      2 1.e3
      2 1.e-20
      2 1.0e-10
      1 8.24e-22
      1 5.e3
      1 4.9e-4
      1 3.34e-27
      1 2.827e-10
      1 1.e99
      1 1.e-8
      1 1e-38
      1 1e-304
      1 1e-30
      1 1e-3
      1 1.e-3
      1 1.e-24
```

**002**

```
$ grep -rnE '1\.?e-30|1\.e9|1e-6|1\.e-6|1\.?e-300|1e-304|1\.?0?e-10|1e-7|29\.e-3|1\.e-20|1e-38|1\.e3\b|1\.e99' src --include=*.cpp --include=*.hpp --include=*.h --include=*.cu --include=*.cuh
src/bc/internal_boundary.cpp:32:  op->solid_density() = node["solid-density"].as<double>(1.e3);
src/bc/internal_boundary.cpp:33:  op->solid_pressure() = node["solid-pressure"].as<double>(1.e9);
src/bc/internal_boundary.hpp:50:  ADD_ARG(double, solid_density) = 1.e3;
src/bc/internal_boundary.hpp:51:  ADD_ARG(double, solid_pressure) = 1.e9;
src/coord/coordinate.cpp:37:                           distance.index(hi).clamp_min(1.e-30));
src/coord/coordinate.cpp:50:        dst, (value.index(hi) - value.index(lo)) / denom.clamp_min(1.e-30));
src/coord/coordinate.cpp:57:                           distance.index(hi).clamp_min(1.e-30));
src/coord/coordinate.cpp:628:      rbar > 0.5 * h, h.square() / (12. * rbar.square().clamp_min(1.e-300)),
src/coord/coordinate.cpp:657:               (12. * rbar * (twelve_r2 + h2)).clamp_min(1.e-300);
src/coord/coordinate.cpp:731:  auto curl1 = derivative(v3 * dl3, dx2, DIM2) / dl3.clamp_min(1.e-30) -
src/coord/coordinate.cpp:732:               derivative(v2 * dl2, dx3, DIM3) / dl2.clamp_min(1.e-30);
src/coord/coordinate.cpp:733:  auto curl2 = derivative(v1 * dl1, dx3, DIM3) / dl1.clamp_min(1.e-30) -
src/coord/coordinate.cpp:734:               derivative(v3 * dl3, dx1, DIM1) / dl3.clamp_min(1.e-30);
src/coord/coordinate.cpp:735:  auto curl3 = derivative(v2 * dl2, dx1, DIM1) / dl2.clamp_min(1.e-30) -
src/coord/coordinate.cpp:736:               derivative(v1 * dl1, dx2, DIM2) / dl1.clamp_min(1.e-30);
src/eos/equation_of_state.cpp:102:  op->weight() = node["weight"].as<double>(29.e-3);
src/eos/equation_of_state.cpp:428:    // density is repaired without a redo; kinetics leaves ~1e-304 in a
src/eos/equation_of_state.hpp:49:  ADD_ARG(double, weight) = 29.e-3;  // mean molecular weight in kg/mol
src/eos/equation_of_state.hpp:54:  ADD_ARG(double, density_floor) = 1.e-6;
src/forcing/axial_momentum.cpp:41:                  .slice(1, pg_->jl(), pg_->ju() + 1) < 1.e-20)
src/forcing/axial_momentum.cpp:90:    auto onaxis = (kk < 1.e-20).to(target.options());  // [n3, n2, 1]
src/hydro/balance_column.hpp:20:    bool wall_clamp = true, double rtol = 1.e-10, int max_iter = 120,
src/hydro/flux_positivity.cpp:71:  // finfo(dtype).tiny: 1e-300 was 0 in float32 (only where() discards it)
src/hydro/hydro.cpp:350:  auto dt_min = torch::tensor({1.e9, 1.e9, 1.e9},
src/hydro/hydro.cpp:385:        auto prod = (lo.abs() * hi.abs()).clamp_min(1.e-30);
src/hydro/hydro.cpp:387:        return torch::where((hi - lo).abs() >= csf, dts, 1.e9)
src/hydro/hydro_ref_x1_impl.h:235:        fabs(ratio - T(1)) < T(1.e-6) ? T(0.5) * (lo + hi) : dp / log(ratio);
src/implicit/forward_sweep_impl.h:58:            std::fmax(c[i].row(r).cwiseAbs().maxCoeff(), T(1e-30))))));
src/implicit/implicit_hydro.cpp:547:    // Keep the double floor unchanged; 1e-300 is zero in float32.
src/implicit/implicit_hydro.cpp:550:                             : 1.e-300;
src/mesh/mesh.cpp:344:  double dt_local = 1.e99;
src/mesh/meshblock_advance.cpp:76:  double dt_min = 1.e9;
src/output/netcdf.cpp:40:// NC_FLOAT keeps ~7 significant digits: a relative change below ~1e-7 in a
src/output/netcdf.cpp:41:// field, or a time step below ~1e-7 * t, does not show in the file.
src/recon/recon_point.h:88:  T alpha0 = T(2.0 / 3.0) / SNAP_SQR(beta0 + T(1e-6));
src/recon/recon_point.h:89:  T alpha1 = T(1.0 / 3.0) / SNAP_SQR(beta1 + T(1e-6));
src/recon/recon_point.h:135:  T alpha0 = d[0] / SNAP_SQR(beta0 + T(1e-6));
src/recon/recon_point.h:136:  T alpha1 = d[1] / SNAP_SQR(beta1 + T(1e-6));
src/recon/recon_point.h:137:  T alpha2 = d[2] / SNAP_SQR(beta2 + T(1e-6));
src/recon/reconstruct.cpp:138:    // dw2 > 0 keeps dwl + dwr away from zero; an epsilon cannot (0/0 at 1e-38)
… trimmed: 45 output lines in total, the first 40 shown
```

**003**

```
$ grep -n 'TINY_NUMBER\|^template\|^inline\|^SNAP_\|DISPATCH_MACRO\|^[A-Za-z].*($' src/riemann/riemann_kernels.h | head -80
29:template <typename T>
30:void DISPATCH_MACRO hllc_impl(T* flx, T* wl, T* wr, T el, T er, T gammal,
41:  constexpr T TINY_NUMBER = T(1.0e-10);
79:  auto bp = ar > T(0.0) ? ar : (TINY_NUMBER);
80:  auto bm = al < T(0.0) ? al : -(TINY_NUMBER);
113:    vxl = al < T(0) ? cl * ql : WL(ivx) + TINY_NUMBER;
114:    vxr = ar > T(0) ? -cr * qr : WR(ivx) - TINY_NUMBER;
188:template <typename T>
189:void DISPATCH_MACRO lmars_impl(T* flx, T* wl, T* wr, T hl, T hr, T gammal,
267:template <typename T>
268:inline DISPATCH_MACRO T roe_max(T a, T b) {
272:template <typename T>
273:void DISPATCH_MACRO roe_impl(T* flx, T const* wl, T const* wr, T el, T er,
279:  auto TINY_NUMBER = T(1.0e-10);
366:  T cs_sq = roe_max(gm1_roe * roe_max(q, TINY_NUMBER), TINY_NUMBER);
476:template <typename T>
477:void DISPATCH_MACRO shallow_roe_impl(T* flx, T const* wl, T const* wr, int dim,
545:template <typename T>
559:template <RiemannKind K, typename T>
560:DISPATCH_MACRO void riemann_central(T* cen, T const* wl, T const* wr,
584:template <RiemannKind K, bool C, bool E, typename T>
585:DISPATCH_MACRO void riemann_face(T* const* p, RiemannArgs<T> const& a) {
```

**004**

```
$ grep -rn 'ivy = IVX + ((ivx - IVX) + 1) % 3\|IPR - dim\|IPR - a.dim\|IPR - dim_\|ivx = IPR -' src | grep -v '^src/riemann/riemann_kernels.h:[0-9]*: *//'
src/riemann/riemann_kernels.h:43:  auto ivx = IPR - dim;
src/riemann/riemann_kernels.h:44:  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
src/riemann/riemann_kernels.h:192:  auto ivx = IPR - dim;
src/riemann/riemann_kernels.h:193:  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
src/riemann/riemann_kernels.h:281:  auto ivx = IPR - dim;
src/riemann/riemann_kernels.h:282:  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
src/riemann/riemann_kernels.h:568:    int ivx = IPR - a.dim;
```

**005**

```
$ grep -rnE 'SQR\(x\)|define SNAP_SQR|define SQR' src
src/implicit/flux_decomposition_impl.h:12:#define SQR(x) ((x) * (x))
src/recon/recon_point.h:12:#define SNAP_SQR(x) ((x) * (x))
src/riemann/riemann_kernels.h:22:#define SQR(x) ((x) * (x))
src/riemann/riemann_kernels.h:184:#define SQR(x) ((x) * (x))
src/riemann/riemann_kernels.h:260:#define SQR(x) ((x) * (x))
```

**006**

```
$ grep -rnE '% ?3\b' src --include=*.cpp --include=*.hpp --include=*.h --include=*.cu --include=*.cuh | grep -iE 'iv|dim|vel' 
src/forcing/diffusion_cubed_sphere.cpp:246:      reciprocal[j] = torch::cross(deriv[(j + 1) % 3].narrow(0, 4, 3),
src/forcing/diffusion_cubed_sphere.cpp:247:                                   deriv[(j + 2) % 3].narrow(0, 4, 3), 0);
src/implicit/flux_decomposition_impl.h:82:  T v = prim[IVX + (IVY - IVX + dir) % 3];
src/implicit/flux_decomposition_impl.h:83:  T w = prim[IVX + (IVZ - IVX + dir) % 3];
src/implicit/flux_decomposition_impl.h:112:  T v2 = w[IVX + (IVY - IVX + dir) % 3];
src/implicit/flux_decomposition_impl.h:113:  T v3 = w[IVX + (IVZ - IVX + dir) % 3];
src/implicit/forward_sweep_impl.h:71:      rhs(2) = DU(IVX + (IVY - IVX + dir) % 3, i) / dt;
src/implicit/forward_sweep_impl.h:72:      rhs(3) = DU(IVX + (IVZ - IVX + dir) % 3, i) / dt;
src/implicit/vic_assemble_impl.h:174:    int const it[2] = {IVX + (dir + 1) % 3, IVX + (dir + 2) % 3};
src/implicit/vic_redistribute_impl.h:83://   MASS(IVX+(dir+1)%3, i) : the (clamped) DRY-gas transfer through that face
src/implicit/vic_redistribute_impl.h:88://   MASS(IVX+(dir+2)%3, i) : the TOTAL (dry + species) mass that face moved
src/implicit/vic_redistribute_impl.h:143:    MASS(IVX + (dir + 1) % 3, i + 1) = q;
src/implicit/vic_redistribute_impl.h:144:    MASS(IVX + (dir + 2) % 3, i + 1) = q;
src/implicit/vic_redistribute_impl.h:172:      MASS(IVX + (dir + 2) % 3, i + 1) += q;
src/implicit/vic_redistribute_impl.h:184:// MASS(IVX+(dir+2)%3, i+1) of the face between cells i and i+1; must run
src/implicit/vic_redistribute_impl.h:191:    T q = MASS(IVX + (dir + 2) % 3, i + 1);
src/implicit/vic_redistribute_impl.h:194:      int m = IVX + (dir + t) % 3;
src/implicit/vic_redistribute_impl.h:217:    DU(IVX + (IVY - IVX + dir) % 3, i) = delta[i](2);
src/implicit/vic_redistribute_impl.h:218:    DU(IVX + (IVZ - IVX + dir) % 3, i) = delta[i](3);
src/mesh/meshblock_advance.cpp:189:      auto M = mass_corr[IVY];  // slot IVX+(dir+1)%3 with dir == 0 (x1 only)
src/riemann/riemann_kernels.h:44:  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
src/riemann/riemann_kernels.h:45:  auto ivz = IVX + ((ivx - IVX) + 2) % 3;
src/riemann/riemann_kernels.h:193:  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
src/riemann/riemann_kernels.h:194:  auto ivz = IVX + ((ivx - IVX) + 2) % 3;
src/riemann/riemann_kernels.h:282:  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
src/riemann/riemann_kernels.h:283:  auto ivz = IVX + ((ivx - IVX) + 2) % 3;
```

**007**

```
$ grep -rnE 'compute\("(W->I|W->A|WA->L|W->L|WL->A|W->T|W->H|W->S|U->W|W->U|DT->|DPY|UT->|W->C|W->E)' src | sort
src/bc/bc_func.cpp:156:    auto a = op.eos->compute("W->A", {t});
src/bc/bc_func.cpp:157:    auto c = op.eos->compute("WA->L", {t, a});
src/chemistry/chemistry_source.cpp:39:  auto temp = peos->compute("W->T", {hydro_w});
src/eos/aneos.cpp:100:  auto [P, U, L] = pthermo->compute("DT->PUL", {cons[IDN], temp});
src/eos/equation_of_state.cpp:333:    auto min_ie = compute("UT->I", {cons_t, min_temp});
src/forcing/diffusion.cpp:732:      auto temp = phydro->peos->compute("W->T", {w});
src/forcing/diffusion_cubed_sphere.cpp:377:  auto temp = s.temp.defined() ? s.temp : phydro->peos->compute("W->T", {s.w});
src/forcing/diffusion_radial.cpp:45:  auto w = phydro->peos->compute("U->W", {u.clone()});
src/forcing/diffusion_radial.cpp:46:  auto alltemp = phydro->peos->compute("W->T", {w});
src/hydro/balance_column.cpp:92:      auto u = eos->compute("W->U", {wp});
src/hydro/balance_column.cpp:93:      auto w4 = eos->compute("U->W", {u.clone()});
src/hydro/hydro.cpp:340:    cs = peos->compute("W->L", {w});
src/hydro/hydro.cpp:342:    auto gamma = peos->compute("W->A", {w});
src/hydro/hydro.cpp:343:    cs = peos->compute("WA->L", {w, gamma});
src/hydro/hydro.cpp:474:      auto cs = peos->compute("W->L", {w});
src/hydro/hydro.cpp:475:      return peos->compute("WL->A", {w, cs});
src/hydro/hydro.cpp:477:    return peos->compute("W->A", {state});
src/hydro/hydro_forward.cpp:92:  auto enth = peos->compute("W->I", {wbar}) + p;
src/hydro/hydro_forward.cpp:981:  auto temp = peos->compute("W->T", {w});
src/mesh/meshblock.cpp:334:  auto u_ref = phydro->peos->compute("W->U", {w_ref});
src/mesh/meshblock.cpp:355:  vars["hydro_u"] = phydro->peos->compute("W->U", {hydro_w});
src/mesh/meshblock_exchange.cpp:212:    auto u = phydro->peos->compute("W->U", {w});
src/mesh/meshblock_restart.cpp:184:      tmp["hydro_w"] = phydro->peos->compute("U->W", {hydro_u.clone()});
src/output/load_diag_output_data.cpp:63:    auto temp = peos->compute("W->T", {w});
src/recon/conversion.cpp:90:  eos.compute("U->W", {tmp, w});
src/riemann/riemann.cpp:181:    auto csl = peos->compute("W->L", {wl});
src/riemann/riemann.cpp:182:    auto csr = peos->compute("W->L", {wr});
src/riemann/riemann.cpp:251:      elr[side] = peos->compute("W->I", {w});
src/riemann/riemann.cpp:253:        clr[side] = peos->compute("W->L", {w});
src/riemann/riemann.cpp:254:        glr[side] = peos->compute("WL->A", {w, clr[side]});
src/riemann/riemann.cpp:256:        glr[side] = peos->compute("W->A", {w});
src/riemann/riemann.cpp:257:        clr[side] = peos->compute("WA->L", {w, glr[side]});
src/riemann/riemann.cpp:263:      elr[side] = peos->compute("W->I", {w}) / w[IDN];
src/riemann/riemann.cpp:265:        clr[side] = peos->compute("W->L", {w});
src/riemann/riemann.cpp:266:        glr[side] = peos->compute("WL->A", {w, clr[side]});
src/riemann/riemann.cpp:268:        glr[side] = peos->compute("W->A", {w});
src/riemann/riemann.cpp:284:    auto eil = peos->compute("W->I", {wl});
src/riemann/riemann.cpp:285:    auto eir = peos->compute("W->I", {wr});
src/riemann/riemann.cpp:286:    auto gammal = peos->compute("W->A", {wl});
src/riemann/riemann.cpp:287:    auto gammar = peos->compute("W->A", {wr});
… trimmed: 44 output lines in total, the first 40 shown
```

**008**

```
$ grep -rnE '\.as<[^>]+>\(("[^"]*"|-?[0-9][0-9.e+-]*|true|false)\)' src | awk -F/ '{print $2}' | sort | uniq -c
      3 bc
     12 coord
      5 eos
      6 forcing
      2 implicit
      4 layout
      4 mesh
      3 recon
      4 scalar
```

**009**

```
$ grep -rnE '\.as<[^>]+>\(("[^"]*"|-?[0-9][0-9.e+-]*|true|false)\)' src/layout src/coord src/output src/eos src/recon src/riemann src/diagnostics src/scalar src/utils src/math
src/layout/layout.cpp:269:  op->type() = node["layout"].as<std::string>("slab");
src/layout/layout.cpp:270:  op->py(node["nb3"].as<int>(1));
src/layout/layout.cpp:271:  op->px(node["nb2"].as<int>(1));
src/layout/layout.cpp:272:  op->pz(node["nb1"].as<int>(1));
src/coord/coordinate.cpp:71:  op->type(node["type"].as<std::string>("cartesian"));
src/coord/coordinate.cpp:79:    x1min = node["bounds"]["x1min"].as<double>(0.0);
src/coord/coordinate.cpp:84:      x2min = node["bounds"]["x2min"].as<double>(0.0);
src/coord/coordinate.cpp:90:      x3min = node["bounds"]["x3min"].as<double>(0.0);
src/coord/coordinate.cpp:93:    x1max = node["bounds"]["x1max"].as<double>(1.0);
src/coord/coordinate.cpp:98:      x2max = node["bounds"]["x2max"].as<double>(1.0);
src/coord/coordinate.cpp:104:      x3max = node["bounds"]["x3max"].as<double>(1.0);
src/coord/coordinate.cpp:128:  op->global_nx1() = node["cells"]["nx1"].as<int>(1);
src/coord/coordinate.cpp:129:  op->global_nx2() = node["cells"]["nx2"].as<int>(1);
src/coord/coordinate.cpp:130:  op->global_nx3() = node["cells"]["nx3"].as<int>(1);
src/coord/coordinate.cpp:213:  op->nghost() = node["cells"]["nghost"].as<int>(1);
src/coord/coordinate.cpp:214:  op->interp_order() = node["cells"]["interp_order"].as<int>(2);
src/eos/equation_of_state.cpp:96:  op->type() = node["type"].as<std::string>("moist-mixture");
src/eos/equation_of_state.cpp:101:  op->gammad() = node["gammad"].as<double>(1.4);
src/eos/equation_of_state.cpp:102:  op->weight() = node["weight"].as<double>(29.e-3);
src/eos/equation_of_state.cpp:115:  op->limiter() = node["limiter"].as<bool>(false);
src/eos/equation_of_state.cpp:121:  op->eos_file() = node["eos-file"].as<std::string>("");
src/recon/reconstruct.cpp:205:  op->shock() = node[section]["shock"].as<bool>(false);
src/recon/reconstruct.cpp:207:  op->interp()->type() = node[section]["type"].as<std::string>("dc");
src/recon/reconstruct.cpp:208:  op->interp()->scale() = node[section]["scale"].as<bool>(false);
src/scalar/scalar_options.cpp:26:  op->nvar() = node["nvar"].as<int>(0);
src/scalar/scalar_options.cpp:27:  op->upper_bound() = node["upper-bound"].as<double>(-1.);
src/scalar/scalar_options.cpp:37:    op->recon()->shock() = recon["shock"].as<bool>(false);
src/scalar/scalar_options.cpp:38:    op->recon()->interp()->type() = recon["type"].as<std::string>("dc");
```

**010**

```
$ grep -rnE 'options->(px|py|pz)\(\)|->(px|py|pz)\(\)|nb[123]"' src python/csrc 2>/dev/null
src/coord/coordinate.cpp:188:  int nb1 = playout->options->pz();
src/coord/coordinate.cpp:189:  int nb2 = playout->options->px();
src/coord/coordinate.cpp:190:  int nb3 = playout->options->py();
src/coord/coordinate.cpp:348:  int nb1 = layout->pz();
src/coord/coordinate.cpp:349:  int nb2 = layout->px();
src/coord/coordinate.cpp:350:  int nb3 = layout->py();
src/coord/gnomonic_equiangle.cpp:234:  N = op->nx2() * pmb->options->layout()->px();
src/eos/equation_of_state.cpp:388:          count > 0 && whole_column && layout && layout->options->pz() > 1;
src/forcing/diffusion.cpp:437:            phydro->pmb->options->layout()->pz() == 1,
src/forcing/diffusion.cpp:466:  int nb1 = layout ? layout->pz() : 1;
src/hydro/hydro_forward.cpp:592:    if (!options->disable_flux_x1() && playout && playout->options->pz() > 1) {
src/layout/cubed_layout.cpp:12:  int px = options->px();
src/layout/cubed_layout.cpp:13:  int py = options->py();
src/layout/cubed_layout.cpp:14:  int pz = options->pz();
src/layout/cubed_layout.cpp:28:  for (int r = 0; r < options->px() * options->py() * options->pz(); ++r) {
src/layout/cubed_layout.cpp:35:  if (rank < 0 || rank >= options->px() * options->py() * options->pz())
src/layout/cubed_layout.cpp:52:      nx += options->px();
src/layout/cubed_layout.cpp:53:    else if (nx >= options->px())
src/layout/cubed_layout.cpp:54:      nx -= options->px();
src/layout/cubed_layout.cpp:56:    if (nx < 0 || nx >= options->px()) return -1;
src/layout/cubed_layout.cpp:61:      ny += options->py();
src/layout/cubed_layout.cpp:62:    else if (ny >= options->py())
src/layout/cubed_layout.cpp:63:      ny -= options->py();
src/layout/cubed_layout.cpp:65:    if (ny < 0 || ny >= options->py()) return -1;
src/layout/cubed_layout.cpp:70:      nz += options->pz();
src/layout/cubed_layout.cpp:71:    else if (nz >= options->pz())
src/layout/cubed_layout.cpp:72:      nz -= options->pz();
src/layout/cubed_layout.cpp:74:    if (nz < 0 || nz >= options->pz()) return -1;
src/layout/cubed_layout.cpp:77:  return _rankof[linear_index3(options->px(), options->py(), nz, ny, nx)];
src/layout/cubed_sphere_layout.cpp:428:  TORCH_CHECK(options->pz() == 1,
src/layout/cubed_sphere_layout.cpp:431:      options->px() == options->py(),
src/layout/cubed_sphere_layout.cpp:434:  int px = options->px();
src/layout/cubed_sphere_layout.cpp:435:  int py = options->py();
src/layout/cubed_sphere_layout.hpp:34:  int pxy() const { return options->px(); }
src/layout/exchange.cpp:53:  bool wrap = (dx != 0 && (rx + dx < 0 || rx + dx >= layout->options->px())) ||
src/layout/exchange.cpp:54:              (dy != 0 && (ry + dy < 0 || ry + dy >= layout->options->py()));
src/layout/exchange.cpp:60:  if (!layout || layout->options->periodic_z() || layout->options->pz() <= 1)
src/layout/exchange.cpp:125:  if (!layout || layout->options->pz() <= 1) return {-1, -1};
src/layout/layout.cpp:266:             {"layout", "nb1", "nb2", "nb3", "verbose", "blocks_per_process",
src/layout/layout.cpp:270:  op->py(node["nb3"].as<int>(1));
… trimmed: 78 output lines in total, the first 40 shown
```

**011**

```
$ grep -rnE 'rbar|x1f.pow\(3\)|pow\(3\) */ *3|rp\.pow|rm\.pow|\(rp - rm\)|rp \* rp|x1f\.square\(\)|x1v\.square|\.pow\(2\) */ *2' src | grep -v '^\s*//' 
src/coord/coordinate.cpp:528:  auto v = x1f.pow(3) / 3.;
src/coord/coordinate.cpp:615:  return dx1f.square() / 12.;
src/coord/coordinate.cpp:623:  auto rbar = 0.5 * (rm + rp);
src/coord/coordinate.cpp:624:  // rbar <= h/2 is a degenerate face (an x1 ghost that reaches r <= 0): the
src/coord/coordinate.cpp:628:      rbar > 0.5 * h, h.square() / (12. * rbar.square().clamp_min(1.e-300)),
src/coord/coordinate.cpp:644:  auto rbar = 0.5 * (rm + rp);
src/coord/coordinate.cpp:648:  //       = 3 rbar (4 rbar^2 + h^2)/(12 rbar^2 + h^2)
src/coord/coordinate.cpp:649:  //   r_c = (2/3)(rp^3-rm^3)/(rp^2-rm^2) = (12 rbar^2 + h^2)/(12 rbar)
src/coord/coordinate.cpp:650:  // Both are ~rbar, so subtracting them directly would cancel. Doing the
src/coord/coordinate.cpp:652:  //   r_v - r_c = h^2 (12 rbar^2 - h^2) / (12 rbar (12 rbar^2 + h^2))
src/coord/coordinate.cpp:653:  //             = h^2/(12 rbar) + O(h^4/rbar^3)
src/coord/coordinate.cpp:655:  auto twelve_r2 = 12. * rbar.square();
src/coord/coordinate.cpp:657:               (12. * rbar * (twelve_r2 + h2)).clamp_min(1.e-300);
src/coord/coordinate.cpp:659:  return torch::where(rbar > 0.5 * h, shift, torch::zeros_like(h));
src/coord/coordinate.hpp:288:   * denominator of the average. With `h = rp - rm` and `rbar = (rm+rp)/2`,
src/coord/coordinate.hpp:290:   *   r_c      = (3 rbar^2 + h^2/4) / (3 rbar)
src/coord/coordinate.hpp:291:   *   sigma1^2 = <r^2> - r_c^2 = (h^2/12) (1 - h^2/(12 rbar^2)) .
src/coord/coordinate.hpp:293:   * metric only shifts the Cartesian `h^2/12` by a relative `h^2/(12 rbar^2)`
src/coord/coordinate.hpp:294:   * and recovers it exactly as `h/rbar -> 0`.
src/coord/coordinate.hpp:320:  //! cancellation-free: `h^2 (12 rbar^2 - h^2) / (12 rbar (12 rbar^2 + h^2))`.
src/coord/gnomonic_equiangle.cpp:291:      (dx1f * (rp * rp + rp * rm + rm * rm) / 3.0).unsqueeze(0).unsqueeze(1);
src/coord/spherical_polar.cpp:20:  return 0.75 * (rp.pow(4) - rm.pow(4)) / (rp.pow(3) - rm.pow(3));
src/coord/spherical_polar.cpp:80:  auto radial_volume = (rp.pow(3) - rm.pow(3)) / 3.0;
src/coord/spherical_polar.cpp:81:  auto radial_area23 = 0.5 * (rp.pow(2) - rm.pow(2));
src/coord/spherical_polar.cpp:134:  return x1f.square().unsqueeze(0).unsqueeze(1) *
src/coord/x1_centroid.hpp:9://! average, located at x1v = rbar + O(dx1^2 / r), while every x1 cell-to-face
src/forcing/axial_momentum.cpp:152:  auto reff = 2. * (rp * rp * rp - rm * rm * rm) / (3. * (rp * rp - rm * rm));
```

**012**

```
$ grep -rnE '== "cubed-sphere"' src | grep -v '^src/layout/layout.cpp'
src/coord/coordinate.cpp:186:  if (playout->options->type() == "cubed-sphere") lx1 = 0;
src/coord/coordinate.cpp:346:  if (layout->type() == "cubed-sphere") lx1 = 0;
src/coord/gnomonic_equiangle.cpp:79:  auto face_id = pmb->options->layout()->type() == "cubed-sphere"
src/forcing/diffusion.cpp:436:        phydro->pmb->options->layout()->type() == "cubed-sphere" &&
src/hydro/hydro_forward.cpp:706:    if (playout->options->type() == "cubed-sphere") {
src/hydro/hydro_forward.cpp:718:    if (playout->options->type() == "cubed-sphere") {
src/hydro/hydro_forward.cpp:725:  if (playout->options->type() == "cubed-sphere") {
src/hydro/hydro_forward.cpp:863:      if (playout && playout->options->type() == "cubed-sphere") {
src/layout/exchange.cpp:47:  if (layout->options->type() == "cubed-sphere" &&
src/mesh/meshblock.cpp:121:  if (options->layout()->type() == "cubed-sphere") {
src/mesh/meshblock_exchange.cpp:40:      !opts.cross_panel_only() && lo.type() == "cubed-sphere" &&
src/mesh/meshblock_exchange.cpp:119:      if (options->layout()->type() == "cubed-sphere" && face >= 2 &&
src/output/netcdf.cpp:105:  bool cs = layout->options->type() == "cubed-sphere";
src/output/writer_core.cpp:269:  if (layout->options->type() == "cubed-sphere") {
src/scalar/scalar.cpp:125:  bool cubed = pmb->get_layout()->options->type() == "cubed-sphere";
```

**013**

```
$ grep -rnE 'std::get<2>\(.*loc_of' src
src/coord/gnomonic_equiangle.cpp:80:                     ? std::get<2>(pmb->get_layout()->loc_of(rank))
src/eos/equation_of_state.cpp:415:        int rz = std::get<2>(layout->loc_of(layout->options->rank()));
src/forcing/coriolis.cpp:70:    auto face = std::get<2>(layout->loc_of(r));
src/layout/cubed_sphere_layout.cpp:187:  int face_id = std::get<2>(loc_of(options->rank()));
src/layout/cubed_sphere_layout.cpp:586:      if (std::get<2>(iloc) != std::get<2>(loc_of(nb))) return;
src/layout/cubed_sphere_layout.cpp:637:    if (std::get<2>(iloc) == std::get<2>(loc_of(nb))) return;
src/layout/cubed_sphere_layout.cpp:764:      if (std::get<2>(iloc) != std::get<2>(loc_of(nb))) return;
src/layout/cubed_sphere_layout.cpp:792:    if (std::get<2>(iloc) == std::get<2>(loc_of(nb))) return;
src/layout/exchange.cpp:48:      std::get<2>(layout->loc_of(nb)) != std::get<2>(iloc))
src/layout/exchange.cpp:205:    if (cubed && std::get<2>(layout->loc_of(sd.peer)) != panel) {
src/layout/layout.cpp:221:  bool same_panel = std::get<2>(iloc) == std::get<2>(loc_of(nb));
src/layout/seam_register.cpp:53:    if (nb < 0 || std::get<2>(layout->loc_of(nb)) == panel) continue;
```

**014**

```
$ grep -rnE 'CS_CART_TO_LOCAL_VEL|CS_LOCAL_TO_CART_VEL|populate_cs_l2g_vel|cs_cart_to_local_vel|cs_local_to_cart_vel|CS_FACE_NAMES|cs_face_xyz_from_tan' src python/csrc
src/coord/cubed_sphere_utils.cpp:23:    if (strcmp(face, CS_FACE_NAMES[n]) == 0) return n;
src/coord/cubed_sphere_utils.cpp:50:  cs_face_xyz_from_tan(cs_face_id(face), a, b, &x, &y, &z);
src/coord/cubed_sphere_utils.cpp:91:      Vec3 v = cs_ab_to_xyz(CS_FACE_NAMES[face_t], alpha_t, beta_t);
src/coord/cubed_sphere_utils.cpp:95:      cs_xyz_to_ab(CS_FACE_NAMES[emap.nface], v, &alpha_s, &beta_s);
src/coord/cubed_sphere_utils.cpp:187:  auto [x, y, z] = cs_ab_to_xyz_tensors(CS_FACE_NAMES[face_id], alpha, beta);
src/coord/cubed_sphere_utils.cpp:205:  auto const g2l = CS_CART_TO_LOCAL_VEL;
src/coord/cubed_sphere_utils.cpp:230:  auto const l2g = CS_LOCAL_TO_CART_VEL;
src/coord/gnomonic_equiangle.cpp:83:      cs_ab_to_lonlat(CS_FACE_NAMES[face_id], mesh[1], mesh[0]);
src/layout/cubed_sphere_constants.h:24:inline DISPATCH_MACRO CSVel cs_cart_to_local_vel(int face, int component) {
src/layout/cubed_sphere_constants.h:33:inline DISPATCH_MACRO CSVel cs_local_to_cart_vel(int face, int component) {
src/layout/cubed_sphere_constants.h:43:inline DISPATCH_MACRO void cs_face_xyz_from_tan(int face, T a, T b, T* x, T* y,
src/layout/cubed_sphere_layout.cpp:21: *   Table `CS_FACE_EDGES[6][4]` and `CS_FACE_NAMES[6][3]` should be
src/layout/cubed_sphere_layout.cpp:287:const char CS_FACE_NAMES[6][3] = {"+X", "+Y", "-X", "+Z", "-Y", "-Z"};
src/layout/cubed_sphere_layout.cpp:289:const CSVel CS_CART_TO_LOCAL_VEL[6][3] = {
src/layout/cubed_sphere_layout.cpp:290:    {cs_cart_to_local_vel(0, VEL1), cs_cart_to_local_vel(0, VEL2),
src/layout/cubed_sphere_layout.cpp:291:     cs_cart_to_local_vel(0, VEL3)},
src/layout/cubed_sphere_layout.cpp:292:    {cs_cart_to_local_vel(1, VEL1), cs_cart_to_local_vel(1, VEL2),
src/layout/cubed_sphere_layout.cpp:293:     cs_cart_to_local_vel(1, VEL3)},
src/layout/cubed_sphere_layout.cpp:294:    {cs_cart_to_local_vel(2, VEL1), cs_cart_to_local_vel(2, VEL2),
src/layout/cubed_sphere_layout.cpp:295:     cs_cart_to_local_vel(2, VEL3)},
src/layout/cubed_sphere_layout.cpp:296:    {cs_cart_to_local_vel(3, VEL1), cs_cart_to_local_vel(3, VEL2),
src/layout/cubed_sphere_layout.cpp:297:     cs_cart_to_local_vel(3, VEL3)},
src/layout/cubed_sphere_layout.cpp:298:    {cs_cart_to_local_vel(4, VEL1), cs_cart_to_local_vel(4, VEL2),
src/layout/cubed_sphere_layout.cpp:299:     cs_cart_to_local_vel(4, VEL3)},
src/layout/cubed_sphere_layout.cpp:300:    {cs_cart_to_local_vel(5, VEL1), cs_cart_to_local_vel(5, VEL2),
src/layout/cubed_sphere_layout.cpp:301:     cs_cart_to_local_vel(5, VEL3)}};
src/layout/cubed_sphere_layout.cpp:303:const CSVel CS_LOCAL_TO_CART_VEL[6][3] = {
src/layout/cubed_sphere_layout.cpp:304:    {cs_local_to_cart_vel(0, VEL1), cs_local_to_cart_vel(0, VEL2),
src/layout/cubed_sphere_layout.cpp:305:     cs_local_to_cart_vel(0, VEL3)},
src/layout/cubed_sphere_layout.cpp:306:    {cs_local_to_cart_vel(1, VEL1), cs_local_to_cart_vel(1, VEL2),
src/layout/cubed_sphere_layout.cpp:307:     cs_local_to_cart_vel(1, VEL3)},
src/layout/cubed_sphere_layout.cpp:308:    {cs_local_to_cart_vel(2, VEL1), cs_local_to_cart_vel(2, VEL2),
src/layout/cubed_sphere_layout.cpp:309:     cs_local_to_cart_vel(2, VEL3)},
src/layout/cubed_sphere_layout.cpp:310:    {cs_local_to_cart_vel(3, VEL1), cs_local_to_cart_vel(3, VEL2),
src/layout/cubed_sphere_layout.cpp:311:     cs_local_to_cart_vel(3, VEL3)},
src/layout/cubed_sphere_layout.cpp:312:    {cs_local_to_cart_vel(4, VEL1), cs_local_to_cart_vel(4, VEL2),
src/layout/cubed_sphere_layout.cpp:313:     cs_local_to_cart_vel(4, VEL3)},
src/layout/cubed_sphere_layout.cpp:314:    {cs_local_to_cart_vel(5, VEL1), cs_local_to_cart_vel(5, VEL2),
src/layout/cubed_sphere_layout.cpp:315:     cs_local_to_cart_vel(5, VEL3)}};
src/layout/cubed_sphere_layout.cpp:354:void populate_cs_l2g_vel(CSVel l2g[6][3]) {
… trimmed: 49 output lines in total, the first 40 shown
```

**015**

```
$ grep -rnE '1\.? *- *[a-z_]*\.square\(\)\)\.sqrt|sqrt\(1\.? *- *[a-z_]+ *\* *[a-z_]+\)|sine_cell_kj|sine_face[23]_kj' src | grep -v 'register_buffer\|^src/coord/gnomonic_equiangle.cpp:(99|102|105)'
src/coord/cartesian.cpp:12:  cosine_cell_kj =
src/coord/cartesian.cpp:16:      "cosine_face2_kj",
src/coord/cartesian.cpp:19:      "cosine_face3_kj",
src/coord/coordinate.cpp:770:  auto c = cosine_cell_kj;
src/coord/coordinate.cpp:771:  auto s = (1. - c.square()).sqrt();
src/coord/coordinate.hpp:169:  torch::Tensor cosine_cell_kj, cosine_face2_kj, cosine_face3_kj;
src/coord/gnomonic_equiangle.cpp:98:  cosine_cell_kj = -x * y / (C * D);
src/coord/gnomonic_equiangle.cpp:99:  sine_cell_kj = (1.0 + x * x + y * y).sqrt() / (C * D);
src/coord/gnomonic_equiangle.cpp:101:  cosine_face2_kj = -xf * y / (Cf * D);
src/coord/gnomonic_equiangle.cpp:102:  sine_face2_kj = (1.0 + xf * xf + y * y).sqrt() / (Cf * D);
src/coord/gnomonic_equiangle.cpp:104:  cosine_face3_kj = -x * yf / (C * Df);
src/coord/gnomonic_equiangle.cpp:105:  sine_face3_kj = (1.0 + x * x + yf * yf).sqrt() / (C * Df);
src/coord/gnomonic_equiangle.cpp:158:  auto fx = face_area2() * sine_face2_kj;
src/coord/gnomonic_equiangle.cpp:159:  auto fy = face_area3() * sine_face3_kj;
src/coord/gnomonic_equiangle.cpp:174:    coord_vec_lower_(aam_kl_c, cosine_cell_kj);
src/coord/gnomonic_equiangle.cpp:333:  auto cos_theta = cosine_face2_kj.narrow(1, 0, options->nc2());
src/coord/gnomonic_equiangle.cpp:334:  auto sin_theta = sine_face2_kj.narrow(1, 0, options->nc2());
src/coord/gnomonic_equiangle.cpp:348:  auto cos_theta = cosine_face3_kj.narrow(0, 0, options->nc3());
src/coord/gnomonic_equiangle.cpp:349:  auto sin_theta = sine_face3_kj.narrow(0, 0, options->nc3());
src/coord/gnomonic_equiangle.cpp:362:  auto cos_theta = cosine_cell_kj;
src/coord/gnomonic_equiangle.cpp:363:  auto sin_theta = sine_cell_kj;
src/coord/gnomonic_equiangle.cpp:411:  auto cos_theta = cosine_cell_kj;
src/coord/gnomonic_equiangle.cpp:412:  auto sin_theta = sine_cell_kj;
src/coord/gnomonic_equiangle.cpp:448:  flux[IVY] = ty + tz * cosine_face2_kj.narrow(1, 0, options->nc2());
src/coord/gnomonic_equiangle.cpp:449:  flux[IVZ] = tz + ty * cosine_face2_kj.narrow(1, 0, options->nc2());
src/coord/gnomonic_equiangle.cpp:477:  flux[IVY] = ty + tz * cosine_face3_kj.narrow(0, 0, options->nc3());
src/coord/gnomonic_equiangle.cpp:478:  flux[IVZ] = tz + ty * cosine_face3_kj.narrow(0, 0, options->nc3());
src/coord/gnomonic_equiangle.cpp:489:  auto cosine = cosine_cell_kj;
src/coord/gnomonic_equiangle.cpp:490:  auto sine2 = sine_cell_kj.square();
src/coord/gnomonic_equiangle.cpp:545:  auto c = cosine_cell_kj;
src/coord/gnomonic_equiangle.hpp:13:  torch::Tensor sine_cell_kj, sine_face2_kj, sine_face3_kj;
src/coord/spherical_polar.cpp:53:      "cosine_cell_kj", torch::zeros({op->nc3(), op->nc2(), 1}, opts));
src/coord/spherical_polar.cpp:55:      "cosine_face2_kj", torch::zeros({op->nc3(), op->nc2() + 1, 1}, opts));
src/coord/spherical_polar.cpp:57:      "cosine_face3_kj", torch::zeros({op->nc3() + 1, op->nc2(), 1}, opts));
src/diagnostics/budget.cpp:320:      coord_vec_raise_(mom, coord->cosine_cell_kj);
src/eos/equation_of_state.cpp:258:  coord_vec_lower_(out, phydro->pmb->pcoord->cosine_cell_kj);
src/eos/equation_of_state.cpp:321:    coord_vec_raise_(mom, pcoord->cosine_cell_kj);
src/eos/ideal_gas.cpp:92:          .add_owned_input(pcoord->cosine_cell_kj.unsqueeze(0).expand_as(prim))
src/eos/ideal_moist.cpp:130:  coord_vec_raise_(raised, phydro->pmb->pcoord->cosine_cell_kj);
src/eos/ideal_moist.cpp:286:  coord_vec_lower_(vel, pcoord->cosine_cell_kj);
… trimmed: 69 output lines in total, the first 40 shown
```

**016**

```
$ grep -rnE 'nvar\(\) - ICY|size\(0\) - ICY|- ICY\b' src | wc -l; grep -rnE 'nvar\(\) - ICY|size\(0\) - ICY|- ICY\b' src | awk -F/ '{print $2}' | sort | uniq -c
31
      3 bc
      1 diagnostics
      1 eos
      4 hydro
      7 implicit
      3 mesh
      4 output
      3 recon
      2 riemann
      2 sedimentation
      1 snap.h:50:
```

**017**

```
$ grep -rnE 'Rgas */|constants::Rgas|8\.314' src python/snapy python/csrc 2>/dev/null
src/eos/ideal_gas.cpp:21:  auto Ri = kintera::constants::Rgas / options->weight();
src/eos/ideal_gas.cpp:40:    auto Rd = kintera::constants::Rgas / options->weight();
src/eos/ideal_moist.cpp:45:  auto Rd = kintera::constants::Rgas * pthermo->inv_mu[0];
src/eos/ideal_moist.cpp:47:    auto Ri = kintera::constants::Rgas * pthermo->inv_mu[i + 1];
src/eos/ideal_moist.cpp:56:  u0 *= kintera::constants::Rgas * pthermo->inv_mu;
src/eos/ideal_moist.cpp:108:                kintera::constants::Rgas * inv_mu;
src/eos/ideal_moist.cpp:110:                    kintera::constants::Rgas * inv_mu;
src/eos/ideal_moist.cpp:111:    split_gas_ = kintera::constants::Rgas * inv_mu;
src/eos/ideal_moist.cpp:256:  auto Rd = kintera::constants::Rgas / options->weight();
src/eos/ideal_moist.cpp:265:                           kintera::constants::Rgas / options->weight();
src/eos/ideal_moist.cpp:276:  auto Rd = kintera::constants::Rgas / options->weight();
src/eos/ideal_moist.cpp:299:  auto Rd = kintera::constants::Rgas / mud;
src/eos/moist_base.cpp:33:    auto Ri = kintera::constants::Rgas * im[n];
src/eos/moist_mixture.cpp:103:  s *= kintera::constants::Rgas * pthermo->inv_mu;
src/eos/moist_mixture.cpp:121:  h *= kintera::constants::Rgas * pthermo->inv_mu;
src/eos/moist_mixture.cpp:155:  ct *= kintera::constants::Rgas * T / dens;
src/output/load_diag_output_data.cpp:100:    auto Rd = kintera::constants::Rgas / thermo_x->mu[0];
```

**018**

```
$ grep -rn 'interp_to_face\|cs_ab_to_lonlat' src python/csrc
src/coord/cubed_sphere_utils.cpp:110:std::pair<torch::Tensor, torch::Tensor> cs_ab_to_lonlat(char const *face,
src/coord/cubed_sphere_utils.hpp:113:std::pair<torch::Tensor, torch::Tensor> cs_ab_to_lonlat(char const *face,
src/coord/gnomonic_equiangle.cpp:83:      cs_ab_to_lonlat(CS_FACE_NAMES[face_id], mesh[1], mesh[0]);
src/output/load_diag_output_data.cpp:22:torch::Tensor interp_to_face(torch::Tensor value, int dim) {
src/output/load_diag_output_data.cpp:142:        auto w1f = interp_to_face(w, 3);
src/output/load_diag_output_data.cpp:147:        auto w2f = interp_to_face(w, 2);
src/output/load_diag_output_data.cpp:152:        auto w3f = interp_to_face(w, 1);
src/output/load_hydro_output_data.cpp:150:      auto [lon, lat] = cs_ab_to_lonlat(face, alpha, beta);
python/csrc/pycoord.cpp:127:      .def("cs_ab_to_lonlat", &snap::cs_ab_to_lonlat)
```

**019**

```
$ grep -rnE 'vapor_ids\(\)\.size\(\) - 1' src
src/eos/equation_of_state.cpp:314:    nvapor = options->thermo()->vapor_ids().size() - 1;
src/eos/ideal_moist.cpp:262:  int nvapor = pthermo->options->vapor_ids().size() - 1;
src/eos/ideal_moist.cpp:273:  int nvapor = pthermo->options->vapor_ids().size() - 1;
src/eos/ideal_moist.cpp:320:  int nvapor = pthermo->options->vapor_ids().size() - 1;
src/riemann/riemann.cpp:277:      params.nvapor = moist->pthermo->options->vapor_ids().size() - 1;
```

**020**

```
$ grep -rnE '0\.5 \* \(.*(IVX|narrow\(0, IVX, 3\)|mom|vel).*\)\.sum\(0\)|0\.5 \* \(prim\.narrow|ke = 0\.5' src
src/diagnostics/budget.cpp:321:      auto ke = 0.5 * (u.narrow(0, IVX, 3) * mom).sum(0, true) / rho;
src/eos/aneos.cpp:55:  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);
src/eos/aneos.cpp:75:  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);
src/eos/equation_of_state.cpp:325:    auto ke = 0.5 * (mom * cons.narrow(0, IVX, 3)).sum(0) / rho;
src/eos/ideal_gas.cpp:70:  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);
src/eos/ideal_gas_impl.h:29:  auto ke = 0.5 * (PRIM(IVX) * CONS(IVX) + PRIM(IVY) * CONS(IVY) +
src/eos/ideal_moist.cpp:131:  auto ke = 0.5 * (momentum * raised).sum(0) / density;
src/eos/ideal_moist.cpp:287:  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * vel).sum(0);
src/eos/moist_base.cpp:93:  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);
src/eos/moist_base.cpp:123:  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);
src/eos/moist_base.cpp:135:  return 0.5 * (cons.narrow(0, IVX, 3) * mom).sum(0) / rho;
src/eos/moist_mixture.cpp:124:  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * vel).sum(0);
src/implicit/flux_decomposition_impl.h:86:  T ke = 0.5 * (u * u + v * v + w * w);
```

**021**

```
$ grep -rnE 'sqrt\(gamma \* pres / dens\)|"WA->L"' src/eos
src/eos/equation_of_state.hpp:143:   *  - "WA->L": compute sound speed from primitive variables and adiabatic
src/eos/ideal_gas.cpp:49:  } else if (ab == "WA->L") {
src/eos/ideal_gas.cpp:54:    return torch::sqrt(gamma * pres / dens);
src/eos/ideal_moist.cpp:195:  } else if (ab == "WA->L") {
src/eos/ideal_moist.cpp:200:    return torch::sqrt(gamma * pres / dens);
src/eos/moist_mixture.cpp:56:  } else if (ab == "WA->L") {
src/eos/moist_mixture.hpp:35:   * Conversions "W->A" and "WA->L" use cached thermodynamic variables for
src/eos/moist_mixture.hpp:42:   * if "WA->L" is needed, it should be preceded mmediately by "W->A".
src/eos/shallow_water.cpp:23:  } else if (ab == "WA->L") {
```

**022**

```
$ grep -rn 'x1_row_faces\|set_x1_faces\|_wl:+\|wr:-' src
src/coord/coordinate.cpp:521:torch::Tensor CoordinateImpl::x1_row_faces(std::string const& interp) const {
src/coord/coordinate.hpp:191:  torch::Tensor x1_row_faces(std::string const& interp) const;
src/hydro/hydro.cpp:55:        !plain ? pmb->pcoord->x1_row_faces(options->recon1()->interp()->type())
src/hydro/hydro.cpp:58:    if (x1f.defined()) precon1->set_x1_faces(x1f);
src/hydro/hydro_forward.cpp:707:      send_vars2["hydro_wl:+"] = wtmp2[ILT];
src/hydro/hydro_forward.cpp:708:      send_vars2["hydro_wr:-"] = wtmp2[IRT];
src/hydro/hydro_forward.cpp:719:      send_vars3["hydro_wl:+"] = wtmp3[ILT];
src/hydro/hydro_forward.cpp:720:      send_vars3["hydro_wr:-"] = wtmp3[IRT];
src/recon/reconstruct.cpp:244:void ReconstructImpl::set_x1_faces(torch::Tensor const& x1f) {
src/recon/reconstruct.hpp:152:  void set_x1_faces(torch::Tensor const& x1f);
src/recon/x1_reference.cpp:193:        auto x1f = !plain               ? pcoord->x1_row_faces(itype)
src/scalar/scalar.cpp:46:      pcoord ? pcoord->x1_row_faces(recon->interp()->type()) : torch::Tensor();
src/scalar/scalar.cpp:49:    precon1->set_x1_faces(x1f);
src/scalar/scalar.cpp:134:    send[d]["scalar_wl:+"] = rlr[d][ILT];
src/scalar/scalar.cpp:135:    send[d]["scalar_wr:-"] = rlr[d][IRT];
```

**023**

```
$ grep -rnoE '0x7[0-9a-fA-F]{3}' src | sort -t: -k3
src/recon/x1_reference.cpp:58:0x7715
src/recon/x1_reference.cpp:59:0x7717
src/hydro/hydro_forward.cpp:596:0x7720
src/hydro/hydro_forward.cpp:655:0x7722
src/hydro/hydro_forward.cpp:477:0x7724
src/hydro/hydro_forward.cpp:1106:0x7726
src/hydro/hydro_forward.cpp:369:0x772a
src/hydro/hydro_forward.cpp:699:0x7730
src/hydro/sweep.hpp:25:0x7730
src/hydro/hydro_forward.cpp:700:0x7734
src/hydro/sweep.hpp:25:0x7734
src/hydro/hydro_forward.cpp:257:0x7740
src/hydro/hydro_forward.cpp:826:0x7750
src/forcing/diffusion_cubed_sphere.cpp:102:0x7760
```

**024**

```
$ grep -rnE 'rfind\("periodic"|"momentum" *\+|"species/"|\.index\(interior\)\.sum\(\{1, 2, 3\}\)|exterior\(false\)\)' src | grep -v '^\s*//'
src/diagnostics/budget.cpp:40:      rows[n] = "momentum" + std::to_string(n - IVX + 1);
src/diagnostics/budget.cpp:44:      rows[n] = "species/" + std::to_string(n - ICY + 1);
src/diagnostics/budget.cpp:61:         names[face].rfind("periodic", 0) != 0;
src/diagnostics/budget.cpp:68:  auto interior = block.part({0, 0, 0}, PartOptions().exterior(false));
src/diagnostics/budget.cpp:70:  return (u * vol).index(interior).sum({1, 2, 3});
src/diagnostics/budget.cpp:309:    auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
src/diagnostics/budget.cpp:322:      add(ke_sum, (ke * vol).index(interior).sum({1, 2, 3}));
src/diagnostics/budget.cpp:338:        add(pe_sum, (pe * vol).index(interior).sum({1, 2, 3}));
src/eos/equation_of_state.cpp:290:  auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
src/eos/equation_of_state.cpp:444:    auto interior = phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false));
src/eos/equation_of_state.cpp:454:          phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false));
src/hydro/flux_positivity.cpp:200:  auto cells = pmb_->part({0, 0, 0}, PartOptions().exterior(false));
src/hydro/flux_positivity.cpp:241:      auto cells = pmb_->part({0, 0, 0}, PartOptions().exterior(false));
src/hydro/hydro_forward.cpp:978:  auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
src/implicit/implicit_hydro.cpp:269:  auto interior = phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false));
src/layout/cubed_sphere_layout.cpp:589:      auto sub = pmb->part(offset, PartOptions().exterior(false));
src/layout/layout.cpp:529:    auto sub = pmb->part(offset, PartOptions().exterior(false));
src/mesh/meshblock_advance.cpp:316:    auto sub = part({0, 0, 0}, PartOptions().exterior(false));
src/mesh/meshblock_redo.cpp:98:  auto interior = part({0, 0, 0}, PartOptions().exterior(false));
src/scalar/scalar.cpp:171:  auto cells = pmb->part({0, 0, 0}, PartOptions().exterior(false));
```

**025**

```
$ grep -rn 'DistributeRows\|GeometryRows\|snap_options/' src python/csrc tests | grep -v '^src/options'
src/forcing/body_heat.cpp:12:#include <snap_options/body_heat_read.inc>
src/forcing/body_heat.cpp:13:#include <snap_options/plume_forcing_read.inc>
src/forcing/coriolis.cpp:19:#include <snap_options/coriolis_read.inc>
src/forcing/diffusion.cpp:25:#include <snap_options/diffusion_read.inc>
src/forcing/forcing.hpp:124:#include <snap_options/coriolis_members.inc>
src/forcing/forcing.hpp:130:#include <snap_options/diffusion_members.inc>
src/forcing/forcing.hpp:143:#include <snap_options/body_heat_members.inc>
src/forcing/forcing.hpp:148:#include <snap_options/top_cool_members.inc>
src/forcing/forcing.hpp:153:#include <snap_options/bot_heat_members.inc>
src/forcing/forcing.hpp:158:#include <snap_options/relax_bot_comp_members.inc>
src/forcing/forcing.hpp:163:#include <snap_options/relax_bot_temp_members.inc>
src/forcing/forcing.hpp:168:#include <snap_options/relax_bot_velo_members.inc>
src/forcing/forcing.hpp:173:#include <snap_options/top_sponge_lyr_members.inc>
src/forcing/forcing.hpp:178:#include <snap_options/bot_sponge_lyr_members.inc>
src/forcing/forcing.hpp:183:#include <snap_options/plume_forcing_members.inc>
src/forcing/layer_heat.cpp:15:#include <snap_options/bot_heat_read.inc>
src/forcing/layer_heat.cpp:16:#include <snap_options/top_cool_read.inc>
src/forcing/relax_bot_comp.cpp:17:#include <snap_options/relax_bot_comp_read.inc>
src/forcing/relax_bot_temp.cpp:12:#include <snap_options/relax_bot_temp_read.inc>
src/forcing/relax_bot_velo.cpp:14:#include <snap_options/relax_bot_velo_read.inc>
src/forcing/sponge.cpp:14:#include <snap_options/bot_sponge_lyr_read.inc>
src/forcing/sponge.cpp:15:#include <snap_options/top_sponge_lyr_read.inc>
src/hydro/hydro.hpp:47:#include <snap_options/hydro_members.inc>
src/hydro/hydro_options.cpp:17:#include <snap_options/hydro_read.inc>
src/output/output_type.cpp:44:#include <snap_options/output_read.inc>
src/output/output_type.hpp:48:#include <snap_options/output_members.inc>
python/csrc/pyhydro.cpp:31:#include <snap_options/hydro_pybind.inc>
python/csrc/pyoutput.cpp:28:#include <snap_options/output_pybind.inc>
tests/test_options_table.cpp:28:#include <snap_options/coord_rows.hpp>
tests/test_options_table.cpp:29:#include <snap_options/internal_boundary_rows.hpp>
tests/test_options_table.cpp:30:#include <snap_options/layout_rows.hpp>
tests/test_options_table.cpp:253:      auto g = GeometryRows::read(config);
tests/test_options_table.cpp:282:    auto d = DistributeRows::read(config);
tests/test_options_table.cpp:314:      GeometryRows::read(YAML::Load(deck));
tests/test_options_table.cpp:331:  auto g = GeometryRows::read(
tests/test_options_table.cpp:343:  EXPECT_EQ(DistributeRows::read(deck).backend(), default_backend());
tests/test_options_table.cpp:344:  EXPECT_EQ(DistributeRows().backend(), default_backend());
tests/test_options_table.cpp:346:  EXPECT_EQ(DistributeRows::read(deck).backend(), "gloo");
```

**026**

```
$ grep -n 'module:\|rows_name:\|struct:' src/options/*.yaml
src/options/coord.yaml:7:module: coord
src/options/coord.yaml:8:rows_name: GeometryRows
src/options/forcing_body_heat.yaml:2:module: body_heat
src/options/forcing_body_heat.yaml:3:struct: BodyHeatOptionsImpl
src/options/forcing_bot_heat.yaml:2:module: bot_heat
src/options/forcing_bot_heat.yaml:3:struct: BotHeatOptionsImpl
src/options/forcing_bot_sponge_lyr.yaml:2:module: bot_sponge_lyr
src/options/forcing_bot_sponge_lyr.yaml:3:struct: BotSpongeLyrOptionsImpl
src/options/forcing_coriolis.yaml:2:module: coriolis
src/options/forcing_coriolis.yaml:3:struct: CoriolisOptionsImpl
src/options/forcing_diffusion.yaml:2:module: diffusion
src/options/forcing_diffusion.yaml:3:struct: DiffusionOptionsImpl
src/options/forcing_plume_forcing.yaml:2:module: plume_forcing
src/options/forcing_plume_forcing.yaml:3:struct: PlumeForcingOptionsImpl
src/options/forcing_relax_bot_comp.yaml:2:module: relax_bot_comp
src/options/forcing_relax_bot_comp.yaml:3:struct: RelaxBotCompOptionsImpl
src/options/forcing_relax_bot_temp.yaml:2:module: relax_bot_temp
src/options/forcing_relax_bot_temp.yaml:3:struct: RelaxBotTempOptionsImpl
src/options/forcing_relax_bot_velo.yaml:2:module: relax_bot_velo
src/options/forcing_relax_bot_velo.yaml:3:struct: RelaxBotVeloOptionsImpl
src/options/forcing_top_cool.yaml:2:module: top_cool
src/options/forcing_top_cool.yaml:3:struct: TopCoolOptionsImpl
src/options/forcing_top_sponge_lyr.yaml:2:module: top_sponge_lyr
src/options/forcing_top_sponge_lyr.yaml:3:struct: TopSpongeLyrOptionsImpl
src/options/hydro.yaml:5:module: hydro
src/options/hydro.yaml:6:struct: HydroOptionsImpl
src/options/internal_boundary.yaml:6:module: internal_boundary
src/options/internal_boundary.yaml:7:rows_name: InternalBoundaryRows
src/options/layout.yaml:5:module: layout
src/options/layout.yaml:6:rows_name: DistributeRows
src/options/output.yaml:6:module: output
src/options/output.yaml:7:struct: OutputOptionsImpl
```

**027**

```
$ grep -rnE 'py::arg\("[a-z_0-9]+"\) *= *[^,)]+' python/csrc | grep -vE '= *(nullptr|py::none\(\)|torch::Tensor\(\))' 
python/csrc/pybc.cpp:68:          py::arg("solid"), py::arg("bfuncs") = std::vector<bcfunc_t>{});
python/csrc/pyeos.cpp:50:                  py::arg("filename"), py::arg("verobse") = false)
python/csrc/pyeos.cpp:84:           py::arg("n") = 0)
python/csrc/pyeos.cpp:86:           py::arg("n") = 0)
python/csrc/pyeos.cpp:90:           py::arg("args") = std::vector<torch::Tensor>{});
python/csrc/pyeos.cpp:127:           py::arg("args") = std::vector<torch::Tensor>{});
python/csrc/pyhydro.cpp:23:                  py::arg("filename"), py::arg("verbose") = false)
python/csrc/pyhydro.cpp:47:        py::arg("grav"), py::arg("wall_clamp") = true, py::arg("rtol") = 1.e-10,
python/csrc/pyhydro.cpp:48:        py::arg("max_iter") = 120, py::arg("geometry") = "",
python/csrc/pyhydro.cpp:49:        py::arg("stretched") = false, py::arg("conversion_order") = 2,
python/csrc/pyimplicit.cpp:25:                  py::arg("filename"), py::arg("verbose") = false)
python/csrc/pymesh.cpp:34:                  py::arg("filename"), py::arg("verbose") = false)
python/csrc/pymesh.cpp:98:          py::arg("name") = "")
python/csrc/pymesh.cpp:124:                  py::arg("filename"), py::arg("verbose") = false)
python/csrc/pymesh.cpp:184:           py::arg("time"), py::arg("final_write") = false)
python/csrc/pymesh.cpp:216:          py::arg("offset"), py::arg("exterior") = true,
python/csrc/pymesh.cpp:217:          py::arg("extend_x1") = 0, py::arg("extend_x2") = 0,
python/csrc/pymesh.cpp:218:          py::arg("extend_x3") = 0)
python/csrc/pymesh.cpp:317:          py::arg("primitive") = false);
python/csrc/pymesh.cpp:358:           py::arg("vars"), py::arg("type") = (int)snap::kConserved,
python/csrc/pymesh.cpp:363:           py::arg("current_time"), py::arg("final_write") = false)
python/csrc/pyoutput.cpp:80:          py::arg("wtflag") = 0);
```

**028**

```
$ grep -nEi 'default|= *(True|False|[0-9.e-]+|"[a-z-]+")' python/snapy/reconstruction.pyi python/snapy/riemann.pyi python/snapy/eos.pyi python/snapy/coordinate.pyi python/snapy/layout.pyi | head -60
python/snapy/reconstruction.pyi:18:        """Initialize InterpOptions with default values."""
python/snapy/reconstruction.pyi:61:        """Initialize ReconstructOptions with default values."""
python/snapy/reconstruction.pyi:95:        """Construct a new default module."""
python/snapy/riemann.pyi:17:        """Initialize RiemannSolverOptions with default values."""
python/snapy/riemann.pyi:41:        """Construct a new default module."""
python/snapy/riemann.pyi:80:        """Construct a new default module."""
python/snapy/riemann.pyi:119:        """Construct a new default module."""
python/snapy/riemann.pyi:158:        """Construct a new default module."""
python/snapy/eos.pyi:18:        """Initialize EquationOfStateOptions with default values."""
python/snapy/eos.pyi:22:    def from_yaml(filename: str, verbose: bool = False) -> "EquationOfStateOptions":
python/snapy/eos.pyi:146:        """Initialize EquationOfState with default values."""
python/snapy/eos.pyi:166:    def species_weight(self, n: int = 0) -> float:
python/snapy/eos.pyi:171:            n: Species index (default 0)
python/snapy/eos.pyi:178:    def species_cv_ref(self, n: int = 0) -> float:
python/snapy/eos.pyi:183:            n: Species index (default 0)
python/snapy/coordinate.pyi:17:        """Initialize CoordinateOptions with default values."""
python/snapy/layout.pyi:16:        """Initialize LayoutOptions with default values."""
python/snapy/layout.pyi:370:        """Initialize CubedSphereLayout with default values."""
```

**029**

```
$ grep -nE '\(("[a-z_-]+"|[0-9.]+e?-?[0-9]*|true|false)\)' python/csrc/pycoord.cpp python/csrc/pylayout.cpp python/csrc/pyrecon.cpp python/csrc/pyriemann.cpp python/csrc/pyscalar.cpp python/csrc/pyeos.cpp python/csrc/pyoutput.cpp | grep -v 'def\("\|def_readwrite\|py::arg\|ADD_OPTION\|\.doc\|R"' | head -30
grep: Unmatched ( or \(
```

**030**

```
$ grep -nE '[(]("[a-z_-]+"|[0-9][0-9.e-]*|true|false)[)]' python/csrc/pycoord.cpp python/csrc/pylayout.cpp python/csrc/pyrecon.cpp python/csrc/pyriemann.cpp python/csrc/pyscalar.cpp python/csrc/pyeos.cpp python/csrc/pyoutput.cpp | grep -vE 'def[(]"|def_readwrite|py::arg|ADD_OPTION' | head -30
python/csrc/pyoutput.cpp:64:            py::object cpp_module = block_obj.attr("cpp_module");
```

**031**

```
$ grep -rnE 'ludcmp|lubksb|partialPivLu|\.lu\(|linalg_solve|torch::linalg::solve|inverse\(\)' src | grep -v '^src/math/'
src/implicit/forward_sweep_impl.h:14:#include <snap/math/lubksb.h>
src/implicit/forward_sweep_impl.h:15:#include <snap/math/ludcmp.h>
src/implicit/forward_sweep_impl.h:49:  // pivot on those rows (FP64 keeps ludcmp's implicit diagonal-block scaling)
src/implicit/forward_sweep_impl.h:82:      if (ludcmp<T, N, !rowscale>(A, indx) == 0)
src/implicit/forward_sweep_impl.h:99:      lubksb(A, indx, solved);
src/implicit/forward_sweep_impl.h:109:      if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
src/implicit/forward_sweep_impl.h:110:      a[i] = a[i].inverse().eval();
src/recon/x1_rows.cpp:276:    auto A = torch::stack(a), Cr = A.narrow(1, 0, 4).inverse();
src/recon/x1_rows.cpp:279:    auto Cp = P.index_select(1, torch::tensor({0, 2, 3})).inverse();
```

**032**

```
$ grep -rln 'TensorCastCache' src | wc -l; grep -rnE 'mutable torch::Tensor .*(cast|_f32|_cached|cache)' src | head
5
```

**033**

```
$ grep -rnE 'OnHost<' src | wc -l; grep -rnE 'thread_local.*Tensor|static std::map<.*Tensor' src | head
10
```

**034**

```
$ grep -rnE 'scalar_type\(\) != .*scalar_type\(\) *\|\|' src | head -30
src/hydro/gravity_work.hpp:204:      cache->gvar.scalar_type() != key.x1f.scalar_type() ||
src/implicit/implicit_hydro.cpp:164:           t.scalar_type() != w.scalar_type() || t.device() != w.device();
src/layout/cubed_sphere_layout.cpp:169:  if (alpha.scalar_type() != velocity.scalar_type() ||
src/layout/cubed_sphere_layout.cpp:174:  if (cosine.defined() && (cosine.scalar_type() != velocity.scalar_type() ||
src/riemann/riemann.cpp:235:      if (buffer->scalar_type() != wl.scalar_type() ||
src/utils/tensor_cast_cache.hpp:32:        snapshot_.scalar_type() != src.scalar_type() ||
src/utils/tensor_cast_cache.hpp:36:        value_.scalar_type() != like.scalar_type() ||
```

**035**

```
$ grep -nE '(rdl|rdr|rd|dryl|dryr) -= ' src/riemann/riemann_kernels.h; grep -nE 'auto ivy = |auto ivz = |int ivx = |auto ivx = ' src/riemann/riemann_kernels.h; grep -nE 'define (SQR|SNAP_SQR)|undef (SQR|SNAP_SQR)' src -r
122:    rdl -= WL(ICY + n);
123:    rdr -= WR(ICY + n);
223:      rd -= WL(ICY + n);
238:      rd -= WR(ICY + n);
303:    dryl -= WL(ICY + n);
304:    dryr -= WR(ICY + n);
573:      rdl -= wl[(ICY + n) * s];
574:      rdr -= wr[(ICY + n) * s];
43:  auto ivx = IPR - dim;
44:  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
45:  auto ivz = IVX + ((ivx - IVX) + 2) % 3;
192:  auto ivx = IPR - dim;
193:  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
194:  auto ivz = IVX + ((ivx - IVX) + 2) % 3;
281:  auto ivx = IPR - dim;
282:  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
283:  auto ivz = IVX + ((ivx - IVX) + 2) % 3;
564:    int ivx = a.dir_yz ? (a.dim == 2 ? IVY : IVZ) : (a.dim == 3 ? IVX : IVY);
568:    int ivx = IPR - a.dim;
src/implicit/flux_decomposition_impl.h:12:#define SQR(x) ((x) * (x))
src/implicit/flux_decomposition_impl.h:130:#undef SQR
src/recon/recon_point.h:12:#define SNAP_SQR(x) ((x) * (x))
src/recon/recon_point.h:163:#undef SNAP_SQR
src/riemann/riemann_kernels.h:22:#define SQR(x) ((x) * (x))
src/riemann/riemann_kernels.h:179:#undef SQR
src/riemann/riemann_kernels.h:184:#define SQR(x) ((x) * (x))
src/riemann/riemann_kernels.h:258:#undef SQR
src/riemann/riemann_kernels.h:260:#define SQR(x) ((x) * (x))
src/riemann/riemann_kernels.h:468:#undef SQR
```

**036**

```
$ grep -nE 'ab == "WA->L"|sqrt\(gamma \* pres / dens\)|auto (dens|pres|gamma) = ' src/eos/ideal_gas.cpp src/eos/ideal_moist.cpp; grep -nE 'Rgas' src/eos/ideal_gas.cpp src/eos/ideal_moist.cpp src/output/load_diag_output_data.cpp
src/eos/ideal_gas.cpp:49:  } else if (ab == "WA->L") {
src/eos/ideal_gas.cpp:51:    auto gamma = args[1];
src/eos/ideal_gas.cpp:52:    auto dens = w[IDN];
src/eos/ideal_gas.cpp:53:    auto pres = w[IPR];
src/eos/ideal_gas.cpp:54:    return torch::sqrt(gamma * pres / dens);
src/eos/ideal_moist.cpp:195:  } else if (ab == "WA->L") {
src/eos/ideal_moist.cpp:197:    auto gamma = args[1];
src/eos/ideal_moist.cpp:198:    auto dens = w[IDN];
src/eos/ideal_moist.cpp:199:    auto pres = w[IPR];
src/eos/ideal_moist.cpp:200:    return torch::sqrt(gamma * pres / dens);
src/eos/ideal_gas.cpp:21:  auto Ri = kintera::constants::Rgas / options->weight();
src/eos/ideal_gas.cpp:40:    auto Rd = kintera::constants::Rgas / options->weight();
src/eos/ideal_moist.cpp:45:  auto Rd = kintera::constants::Rgas * pthermo->inv_mu[0];
src/eos/ideal_moist.cpp:47:    auto Ri = kintera::constants::Rgas * pthermo->inv_mu[i + 1];
src/eos/ideal_moist.cpp:56:  u0 *= kintera::constants::Rgas * pthermo->inv_mu;
src/eos/ideal_moist.cpp:108:                kintera::constants::Rgas * inv_mu;
src/eos/ideal_moist.cpp:110:                    kintera::constants::Rgas * inv_mu;
src/eos/ideal_moist.cpp:111:    split_gas_ = kintera::constants::Rgas * inv_mu;
src/eos/ideal_moist.cpp:256:  auto Rd = kintera::constants::Rgas / options->weight();
src/eos/ideal_moist.cpp:265:                           kintera::constants::Rgas / options->weight();
src/eos/ideal_moist.cpp:276:  auto Rd = kintera::constants::Rgas / options->weight();
src/eos/ideal_moist.cpp:299:  auto Rd = kintera::constants::Rgas / mud;
src/output/load_diag_output_data.cpp:100:    auto Rd = kintera::constants::Rgas / thermo_x->mu[0];
```

**037**

```
$ grep -nE 'T vscale|phi\[k\] = |inline T recon_point' src/recon/recon_point.h; grep -nE 'if \(options->shock\(\)\)|return result;' src/recon/reconstruct.cpp; grep -nE '^bool boundary_face|^torch::Tensor BudgetConsumer::boundary|^void BudgetConsumer::start' src/diagnostics/budget.cpp; grep -nE '^OutputBlockLocation|return loc;' src/output/writer_core.cpp; grep -nE '^RiemannSolverOptions RiemannSolverOptionsImpl::from_yaml|return op;|return from_yaml' src/riemann/riemann.cpp; grep -nE 'cached_geometry_\(|x1_metrics_\(\) const|x1_row_faces\(std' src/coord/coordinate.cpp; grep -n 'scale() = true' src/scalar/scalar.cpp; grep -nE 'bool cubed|pmb->finalize_exchange' src/scalar/scalar.cpp; grep -nE '_cons2ke|return 0.5' src/eos/moist_base.cpp; grep -rln 'SNAP_X1_CENTROID_EXACT' . --include=*.yaml --include=*.cmake --include=*.txt --include=*.yml --include=*.py --include=*.sh | head
70:  T vscale = scale ? (std::fabs(src[0]) + std::fabs(src[stride]) +
79:    phi[k] = src[k * stride] / vscale;
110:  T vscale = scale ? (std::fabs(src[0]) + std::fabs(src[stride]) +
120:    phi[k] = src[k * stride] / vscale;
147:DISPATCH_MACRO inline T recon_point(T const* src, int stride, T const* row,
262:    return result;
282:  if (options->shock()) {
287:    return result;
312:  if (ny <= 0) return result;
319:  return result;
52:bool boundary_face(MeshBlockImpl const& block, int dim, bool upper) {
78:torch::Tensor BudgetConsumer::boundary(MeshBlockImpl const& block,
157:void BudgetConsumer::start(MeshBlockImpl const& block, Variables const& vars) {
260:OutputBlockLocation output_block_location(Layout const& layout, int rank) {
277:  return loc;
53:RiemannSolverOptions RiemannSolverOptionsImpl::from_yaml(
58:  if (!config[section]) return op;
59:  if (!config[section]["riemann-solver"]) return op;
62:  return from_yaml(config["dynamics"]["riemann-solver"]);
65:RiemannSolverOptions RiemannSolverOptionsImpl::from_yaml(
74:  return op;
436:std::array<torch::Tensor, 3> CoordinateImpl::x1_metrics_() const {
521:torch::Tensor CoordinateImpl::x1_row_faces(std::string const& interp) const {
564:torch::Tensor CoordinateImpl::cached_geometry_(
590:  return cached_geometry_(_area1_cache, [&] {
596:  return cached_geometry_(_area2_cache, [&] {
602:  return cached_geometry_(_area3_cache, [&] {
608:  return cached_geometry_(_vol_cache, [&] {
43:  recon->interp()->scale() = true;
125:  bool cubed = pmb->get_layout()->options->type() == "cubed-sphere";
143:      pmb->finalize_exchange(send[d], so.dim(3 - d), works[d]);
64:    *out = _cons2ke(args[0]);
127:torch::Tensor MoistEOSImpl::_cons2ke(torch::Tensor cons) {
135:  return 0.5 * (cons.narrow(0, IVX, 3) * mom).sum(0) / rho;
./docs/derivations/verify_x1_centroid.py
./tests/CMakeLists.txt
./tests/test_radial_order.py
./tests/test_stencil_cache_small_block.py
./tests/test_x1_centroid_rest.py
```

**038**

```
$ grep -n 'SNAP_X1_CENTROID_EXACT' tests/CMakeLists.txt tests/test_radial_order.py tests/test_stencil_cache_small_block.py tests/test_x1_centroid_rest.py | head; grep -lE '^scalar:' $(grep -ohE '[a-zA-Z0-9_./-]+\.yaml' tests/test_radial_order.py tests/test_stencil_cache_small_block.py tests/test_x1_centroid_rest.py | sort -u | sed 's#^#tests/#') 2>&1 | head
tests/CMakeLists.txt:54:                     PROPERTIES ENVIRONMENT "SNAP_X1_CENTROID_EXACT=1")
tests/CMakeLists.txt:125:seam_arm(_x1_centroid "SNAP_X1_CENTROID_EXACT=1"
tests/CMakeLists.txt:130:# the nghost >= 3 check of SNAP_X1_CENTROID_EXACT and of conversion-order 4
tests/CMakeLists.txt:133:seam_arm(_nghost_check_x1_centroid "SNAP_X1_CENTROID_EXACT=1"
tests/CMakeLists.txt:141:  seam_arm(_x1_centroid_cuda "SNAP_X1_CENTROID_EXACT=1"
tests/CMakeLists.txt:160:seam_arm_mp(_x1_centroid "SNAP_X1_CENTROID_EXACT=1")
tests/CMakeLists.txt:167:# SNAP_X1_CENTROID_EXACT set: the same checks hold on their Cartesian columns
tests/CMakeLists.txt:171:                       PROPERTIES ENVIRONMENT "SNAP_X1_CENTROID_EXACT=1")
tests/test_radial_order.py:11:if PLAIN: os.environ["SNAP_X1_CENTROID_EXACT"] = "1"
tests/test_stencil_cache_small_block.py:2:"""SNAP_X1_CENTROID_EXACT on blocks with few x1 cells (#298).
```

**039**

```
$ grep -lnE 'scalar' tests/test_x1_seam_split*.cpp tests/test_stretched_x1_ref.cpp tests/test_balance_column.cpp tests/test_face_floor.cpp tests/test_radial_order.py tests/test_stencil_cache_small_block.py tests/test_x1_centroid_rest.py 2>/dev/null; echo exit=$?
exit=1
```

