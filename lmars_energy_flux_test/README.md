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
