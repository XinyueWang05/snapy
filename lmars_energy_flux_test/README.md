# LMARS energy-flux stability test (snapy a09cc256)

Question (Xi): is the LMARS energy flux unstable under a strong uniform
horizontal wind? At `src/riemann/lmars_impl.h:61/76` (a09cc256) the energy flux
is `F_E = u* (rho h)_upwind`, so the pressure work uses the upwind p, while
the normal-momentum flux (`:58/:73`) uses the face pressure `pbar`. HLLC is the
control.

## Setup

`run_arm.py` builds one arm: 2-D Cartesian, x1 vertical (reflecting), x2
horizontal periodic (64 columns of width dx), no rotation, dry H2-He from
`examples/jupiter_crm_dry.yaml` (cv_R = 2.5, gamma = 1.4, Rd = 3936.04 J/kg/K
read back from the EOS), weno5 (both directions), rk3, cfl 0.3,
`implicit-scheme 9`. The isothermal column is projected onto the discrete
hydrostatic balance with `snapy.balance_column` (10-bar bottom cell), then
1e-6 K Gaussian white noise is added in T (p fixed, seed 20261025) and a
uniform wind u in x2. Every arm runs 3 model days with a diagnostic every
model hour (`series.csv`) and w snapshots daily (`snaps.npz`).

| set | T0 | g | height, levels | H = Rd T0 / g | c_s = sqrt(1.4 Rd T0) | u [m/s] | dx [km] |
|---|---|---|---|---|---|---|---|
| original | 100 K | 8.87 | 450 km, 100 (dz 4.5 km) | 44.37 km (10.14 H) | 742.3 m/s | 0, 50, 250, 400 | 5, 50, 100, 200, 400 |
| hot Jupiter | 1500 K | 23 | 2570 km, 100 (dz 25.7 km) | 256.70 km (10.01 H) | 2875.0 m/s | 0, 1000, 3000, 5000 | 100, 400 |

Mach = |u| / c_s; the hot-Jupiter u = 3000 and 5000 m/s arms are supersonic
(Mach 1.04, 1.74).

Break test: `break_ustar_pbar.patch` (diagnostic only, not a proposed fix)
replaces the LMARS energy flux by `u* (rho e + rho K)_upwind + u* pbar`, i.e.
the same face pressure as the momentum flux. It is built as a second snapy
tree and rerun on the worst lmars arm of each set.

Extra diagnostic (outside the brief): the worst arm of each set with the
vertical solve explicit (`--scheme 0`), 1 model day, both solvers.

## Exact commands

```bash
# environment: python with torch 2.10.0, kintera 2.6.6, pyharp 2.7.6 (pip), numpy, matplotlib
git clone https://github.com/chengcli/snapy src && cd src && git checkout a09cc256 && cd ..
git -C src worktree add ../src-break -b lmars-break-ustar-pbar a09cc256
git -C src-break am ../src/lmars_energy_flux_test/break_ustar_pbar.patch
T=src/lmars_energy_flux_test
$T/build.sh $PWD/src 6
$T/build.sh $PWD/src-break 6
export PYTHON=$(which python) SNAPY_PY=$PWD/src/pyinst
HJ="--T0 1500 --grav 23 --ztop 2570e3"

# u = 0 gates, NO noise (must stay at round-off)
DXS="5 50 400" $T/run_arms.sh runs/gate 6 gate
DXS="400 100"  $T/run_arms.sh runs/hj_gate 4 gate $HJ

# arm matrices, 1e-6 K noise
DXS="5 50 400 100 200" $T/run_arms.sh runs/matrix 6 matrix
DXS="100 400" US="5000 3000 1000 0" $T/run_arms.sh runs/hj 6 matrix $HJ

# break tests (u* pbar build)
(mkdir -p runs/break/lmars_u400_dx5 && cd runs/break/lmars_u400_dx5 && \
 PYTHONPATH=$OLDPWD/src-break/pyinst OMP_NUM_THREADS=1 \
 $PYTHON $OLDPWD/$T/run_arm.py --solver lmars --u 400 --dx 5e3 --out . > run.log 2>&1)
(mkdir -p runs/hj_break/lmars_u5000_dx100 && cd runs/hj_break/lmars_u5000_dx100 && \
 PYTHONPATH=$OLDPWD/src-break/pyinst OMP_NUM_THREADS=1 \
 $PYTHON $OLDPWD/$T/run_arm.py --solver lmars --u 5000 --dx 100e3 --out . $HJ > run.log 2>&1)

# explicit-vertical diagnostic (1 day, both solvers)
$T/run_diag_scheme0.sh runs/diag_scheme0
for s in lmars hllc; do (mkdir -p runs/hj_diag_scheme0/${s}_u5000_dx100 && \
 cd runs/hj_diag_scheme0/${s}_u5000_dx100 && PYTHONPATH=$SNAPY_PY OMP_NUM_THREADS=1 \
 $PYTHON $OLDPWD/$T/run_arm.py --solver $s --u 5000 --dx 100e3 --out . $HJ \
 --days 1 --scheme 0 > run.log 2>&1); done

# tables and figure
for s in gate hj_gate matrix hj break hj_break diag_scheme0 hj_diag_scheme0; do
  $PYTHON $T/analyze.py runs/$s $T/results/$s.md $T/results/$s.csv; done
# growth rate vs |u|/dx and vs Mach, all arms (not committed, see Results)
$PYTHON $T/plot_growth.py growth_vs_u_over_dx_and_mach.png \
  $T/results/matrix.csv $T/results/hj.csv \
  --break $T/results/break.csv --break $T/results/hj_break.csv
```

`results/<set>.md|csv` are the analyze.py tables; `results/series/<set>/<arm>.csv`
are the per-arm rms w(t), max |w|, rms u', rms T' series (copied from
`runs/<set>/<arm>/series.csv`). Decks: `deck_lmars_u400_dx5.yaml` (original),
`deck_hj_lmars_u5000_dx100.yaml` (hot Jupiter); the other arms differ only in
`riemann-solver`, `x2max` (= 64 dx) and the wind/noise set in `run_arm.py`.

## Results

All 70 runs completed 3 model days (1 day for the scheme-0 diagnostic), no
redos, no non-finite states. Runs are deterministic: 8 runs restarted from
t = 0 after a host-session loss reproduced every overlapping diagnostic row of
their partial predecessors exactly.

Gates (u = 0, no noise, 3 d, hourly samples): rms w <= 4.8e-11 m/s (peak in
the first hours at the finest dx), <= 9.1e-12 m/s after day 1, and max |w| <=
9.6e-11 m/s in all 10 gate runs (original: 6, hot Jupiter: 4); hot-Jupiter
rms T' <= 3.65e-13 K (machine epsilon x 1500 K).

Growing arms (everything else: growth 'none', rms w stays at 1e-9..2e-6 m/s):

| set | solver | u | dx | Mach | abs(u)/dx [1/s] | growth [1/s] (e-fold) | lambda/dx at 1 d | rms w 3 d | max abs w |
|---|---|---|---|---|---|---|---|---|---|
| 100 K | lmars | 250 | 5 | 0.34 | 5.0e-2 | 6.14e-5 (4.5 h) | 12.8 | 3.91 | 15.4 |
| 100 K | hllc | 250 | 5 | 0.34 | 5.0e-2 | 6.18e-5 (4.5 h) | 12.8 | 4.31 | 16.9 |
| 100 K | lmars | 400 | 5 | 0.54 | 8.0e-2 | 1.18e-4 (2.3 h) | 12.8 | 4.32 | 86.0 |
| 100 K | hllc | 400 | 5 | 0.54 | 8.0e-2 | 9.27e-5 (3.0 h) | 12.8 | 4.72 | 92.9 |
| 100 K | lmars u* pbar | 400 | 5 | 0.54 | 8.0e-2 | 1.13e-4 (2.5 h) | 12.8 | 5.34 | 78.8 |
| 1500 K | lmars | 3000 | 100 | 1.04 | 3.0e-2 | 9.22e-5 (3.0 h) | 9.1 | 22.4 | 212 |
| 1500 K | hllc | 3000 | 100 | 1.04 | 3.0e-2 | 9.85e-5 (2.8 h) | 9.1 | 35.0 | 336 |
| 1500 K | lmars | 5000 | 100 | 1.74 | 5.0e-2 | 1.71e-4 (1.6 h) | 10.7 | 28.2 | 386 |
| 1500 K | hllc | 5000 | 100 | 1.74 | 5.0e-2 | 1.74e-4 (1.6 h) | 10.7 | 21.2 | 486 |
| 1500 K | lmars u* pbar | 5000 | 100 | 1.74 | 5.0e-2 | 1.73e-4 (1.6 h) | 10.7 | 29.6 | 457 |

Highest non-growing abs(u)/dx: 1.0e-2 1/s (100 K: u = 50, dx = 5 km and
u = 400, dx = 50 km), 1.25e-2 1/s (1500 K: u = 5000, dx = 400 km, Mach 1.74).
Mach number does not separate growing and non-growing arms (Mach 1.74 at
dx = 400 km is stable, Mach 0.34 at dx = 5 km grows); abs(u)/dx does, with a
threshold between 1.25e-2 and 3e-2 1/s in both setups.

Explicit vertical solve (`--scheme 0`, 1 d) on the worst arm of each set:
rms w at 1 d is 6.2e-7 / 6.1e-7 m/s (100 K, lmars / hllc; decaying from
4.9e-6 at 3 h) and 2.55e-7 / 2.55e-7 m/s (1500 K, decaying from 1.45e-6 at
3 h), versus 2.82e-2 / 2.51e-2 and 1.73e-1 / 1.75e-1 m/s with `implicit-scheme 9`.

Conclusion: the growth is not specific to LMARS (HLLC grows at a comparable rate:
within 1-7 % in three of the four growing pairs; lmars 27 % faster at 100 K,
u = 400, dx = 5 km) and is not removed by the consistent u* pbar pressure work
(break test within 5 % of unmodified lmars in both sets). It disappears with
the vertical solve explicit, so the evidence points to the vertically-implicit
scheme under strong horizontal wind, not to the LMARS energy flux. The modes
are smooth, 9-13 dx at 1 d, not few-dx. Because the brief made the growth-rate
figure conditional on the pressure-work term being the cause, the png is not
committed here; `plot_growth.py` regenerates it.

## Courant-number analysis (follow-up, Xi ts 1792989268)

No new runs; `courant.py` re-fits the same arms. For each arm: dt is
`dt_max` from summary.json, the step of the unperturbed flow; it equals
cfl*dx/(|u|+c_s) (horizontal acoustic CFL; the vertical is implicit). Courant
numbers are u*dt/dx and c_s*dt/dz with c_s = 742.3 m/s, dz = 4.5 km
(100 K) and c_s = 2875.0 m/s, dz = 25.7 km (1500 K). Rate: ln(rms w) fit in
1/day. Growing arms are fitted from the minimum after 6 h to the first sample
two decades below saturation. Others are fitted over 1-3 d (0.25-1 d for the
1-day diagnostics) and labelled 'decays' (R^2 >= 0.5) or 'no trend'.

```bash
$PYTHON $T/courant.py runs $T/results/courant.md $T/results/courant.csv \
  matrix hj break hj_break diag_scheme0 hj_diag_scheme0
$PYTHON $T/courant.py runs $T/results/courant_gates.md $T/results/courant_gates.csv gate hj_gate
$PYTHON $T/plot_courant.py $T/results/courant.csv $T/results/growth_vs_courant.png
```

Because dt = cfl*dx/(|u|+c_s), u*dt/dx = cfl*M/(1+M) depends only on the Mach
number, and c_s*dt/dz = cfl*(dx/dz)/(1+M) scales with dx. In these arms dx
and c_s*dt/dz therefore cannot be separated; a dt ladder at fixed dx can.

Growing arms (all others: decay or no trend, -2.9 to +0.01 /day):

| deck | u | dx km | Mach | dt s | u dt/dx | c_s dt/dz | lmars /day (R^2) | hllc /day (R^2) | lambda/dx 1 d |
|---|---|---|---|---|---|---|---|---|---|
| 100 K | 250 | 5 | 0.34 | 1.51 | 0.076 | 0.25 | 5.17 (0.997) | 5.18 (0.997) | 12.8 |
| 100 K | 400 | 5 | 0.54 | 1.31 | 0.105 | 0.22 | 10.6 (0.994) | 10.4 (0.995) | 12.8 |
| 1500 K | 3000 | 100 | 1.04 | 5.11 | 0.153 | 0.57 | 9.14 (0.995) | 9.11 (0.997) | 9.1 |
| 1500 K | 5000 | 100 | 1.74 | 3.81 | 0.191 | 0.43 | 15.2 (0.999) | 15.3 (0.998) | 10.7 |

u* pbar break tests: 10.5 /day (100 K) and 15.3 /day (1500 K). Explicit
vertical (scheme 0, 1 d): decays at -1.8 /day (100 K) and -1.2 /day (1500 K).

- vs u (fixed dx): 100 K dx 5 km: -2.9 (u 50), +5.2 (250), +10.5 (400);
  1500 K dx 100 km: -0.37 (1000), +9.1 (3000), +15.2 (5000).
- vs dx (fixed u): growth only at the finest dx of each deck; 100 K u 400:
  +10.5 at dx 5 km, -0.15 to -0.21 at dx 50-400 km; 1500 K u 5000: +15.2 at
  dx 100 km, about 0 (-0.07/+0.01, R^2 < 0.06) at dx 400 km.
- vs u*dt/dx: does not collapse. The same 0.105 grows at dx 5 km and does not
  at dx 50-400 km, and 0.191 grows at dx 100 km but not at dx 400 km.
- vs c_s*dt/dz: does not collapse the rate, but separates the classes. All arms
  with c_s*dt/dz >= 0.87 (40 arms) do not grow, at any u up to Mach 1.74. All
  growing arms have c_s*dt/dz <= 0.57, and among c_s*dt/dz <= 0.57 only
  u*dt/dx >= 0.076 grows (u = 50, dx = 5 km, 0.019, decays).
- Within the growing arms the rate rises with u*dt/dx (5.2, 10.5, 9.1, 15.2
  /day at 0.076, 0.105, 0.153, 0.191); the 0.153 arm, below trend, has the
  largest c_s*dt/dz (0.57). Neither number alone collapses the rate, and
  rate*dt does not either (1.6e-4 vs 6.7e-4 per step at the two fastest arms).

Hypothesis (not tested by these arms): the growth needs a small vertical
acoustic Courant number and a large horizontal one. A large c_s*dt/dz damps
it, possibly through the implicit solve's damping of vertically propagating
acoustic modes. This would also explain why halving cfl reduces the rate in
Jiheng's leg: it halves u*dt/dx. Separating dt from dx is Jiheng's dt ladder.

GCM regime (scheme 9, u 250/400, dx 50-400 km, 100 K deck): no arm grows.
hllc at u = 250, dx = 400 km (dt 121 s, u*dt/dx 0.076, c_s*dt/dz 19.9):
rms w 3.3e-7 / 3.5e-7 / 3.4e-7 m/s at 1/2/3 d, fit -0.064 /day (R^2 0.11,
no trend), max |w| 4.0e-6 m/s.

Note: the u = 400, dx = 5 km rate here (10.5 /day) is below Jiheng's
16.9 /day for nominally the same arm. That deck (jihenghu/snapy d14b4c9,
results/raw/b_lmars_u400.deck.yaml) uses `ideal-gas` with weight 0.0023
(Rd 3615) and weno5 `shock: false`. This one uses the jupiter_crm_dry
`ideal-moist` EOS (Rd 3936) with `shock: true`. Hypothesis, untested: the
WENO shock flag and/or Rd account for the difference.
