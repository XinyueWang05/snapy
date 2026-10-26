# LMARS strong-wind study: explicit path, uniform-wind column, box budgets

Thread: Xi, ts 1792975246 / 1792995379 (hypotheses H1, H2). Sibling of branch
`lmars-energy-flux-test`, which was not on github.com/XinyueWang05/snapy when
this was set up, so the deck and driver here were rebuilt from the thread spec.

Build: source build of chengcli/snapy at a09cc256c3ab5c2a10315ab4bf32169740c2fd52,
CPU (`cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON`, CUDA off,
then `pip install --no-build-isolation .`), torch 2.10.0, kintera 2.6.6,
Python 3.11. Every run is float64 on CPU, one thread per process.

## Deck (`run_arm.py`, `deck()`; each run's YAML is in `runs/<arm>.yaml`)

- x1 vertical, 0-450 km, 100 levels (dz 4.5 km), reflecting walls; x2 periodic,
  nx2 = 64 (box) or 4 (column); x3 one cell.
- dry H2-He (H 1.5, He 0.15, cv_R 2.5, gamma 1.4, Rd 3936.2 J/kg/K), ideal-moist EOS.
- g 8.87 m/s2, gravity work `face` (default), no rotation, no sponge.
- isothermal T0 100 K, 10 bar at the bottom, balanced with `snapy.balance_column`.
- weno5 (scale, shock) in both directions; lmars; rk3; cfl 0.3.
- uniform wind u0 in x2; isobaric uniform-random T noise of amplitude 1e-6 K
  (seed 20261026). The u = 0 gate has no noise.
- 3 model days; diagnostics every 600 s.
- Column (check a): nx2 = 4 with the same noise in every x2 column, so the state
  stays exactly x2-uniform (x2 flux differences vanish), but dt keeps the x2 limit
  of the implicit schemes. With nx2 = 1 snapy falls back to the acoustic x1 limit.

## Diagnostics (`analyze.py` -> `tables.md`)

- rms w over the interior; growth rate = slope of ln(rms w) vs t (1/day) over rows
  with 1e-4 <= rms w <= 1e-1 ("band"), else over t >= 0.5 d ("run").
- e-fold in units of dx/u = u / (rate * dx).
- dominant wavelength: peak of the x2 power spectrum of w summed over levels,
  in dx, at the first row with rms w >= 1e-2 and at the end.
- max|u-u0|/u0 over the interior; global sums of rho, m1, m2, m3, E and
  E + rho g z (snapy's E has no potential energy).

## Running

    ./make_jobs.sh > jobs.txt
    PY=python NPAR=5 setsid nohup ./launch.sh > launch.log 2>&1 < /dev/null &

`launch.sh` skips arms whose `runs/<arm>.json` exists, so a relaunch resumes;
`runs/<arm>.state` records start/exit for every arm.

## Results (all 34 arms finished 3 model days; full numbers in `tables.md`)

1. Explicit path (implicit-scheme 0, lmars), dx 75/100/125/150 km x u 110/150/250/400
   m/s: no arm grows. dt is the vertical acoustic limit, 1.819 s (cs dt/dz = 0.300,
   u dt/dx = 0.0013-0.0097, Mach 0.148-0.539). rms w decays over 0.5-3 d at
   -0.30 (dx 75), -0.24/-0.25 (dx 100), -0.19/-0.21 (dx 125), -0.15/-0.17 (dx 150)
   per day and ends at 3.8-6.1e-7 m/s. The rate is set by dx, not by u. There is no
   growing mode, so no e-fold or wavelength. The u = 0 no-noise gate ends at
   rms w 6.6e-9 m/s.
2. Uniform-wind column: schemes 0 and 9 keep u0 to 3-5e-12 (scheme 0) and
   3.5-5.1e-13 (scheme 9) relative over 3 d, growing with step count like
   round-off. Scheme 1 departs by 1.8-2.5e-9 relative within the first 600 s
   (while the 1e-6 K noise's acoustic adjustment is moving mass), then holds
   flat. No scheme shows a growth rate in the column (|rate| <= 0.04/d for 1 and 9).
3. Box, u = 400, dx = 100 km: rho and m2 sums drift <= 6.7e-13 (schemes 9 and 1)
   and 8e-12 (scheme 0, 1.4e5 steps); m3 is identically 0 (2-D). E + rho g z drifts
   5.5e-13 (9), 8.0e-12 (0) and 1.5e-8 (1). Scheme 1 grows at 31.4/d
   (band 0.17-0.38 d), e-fold 11.0 dx/u, lambda 6.4 dx, saturating at rms w 0.42
   m/s and rms(u-u0) 20 m/s (13 redos). Scheme 9 decays at -0.25/d like scheme 0.

The 1e-6 K noise and the seed are identical in every arm. The column departure
in scheme 1 scales with u roughly linearly (1.8e-7 m/s at u 100, 9.8e-7 m/s at
u 400 in absolute terms; dt also differs, 35.6 vs 26.3 s).
