# Scheme-1 design checks on ba2b28ce (Xi, ts 1793068432)

Branch `scheme1-design-checks` (based on ba2b28cee183d40a45c066b1d821b632f4b580e3,
the head of `scheme1-donor-momentum`). The experimental code variant for check 1
is on branch `scheme1-roe-vt` (one commit on ba2b28ce). Tables: `tables.md`;
per-run series, profiles, YAML decks, logs and run states: `runs/<label>/`
(final states are summarised by `state_sha256` in each JSON; the .npy files are
not committed).

## Builds and versions

| label | snapy commit | install |
|---|---|---|
| base | a09cc256c3ab5c2a10315ab4bf32169740c2fd52 (2.11.16) | `pip install --no-build-isolation .` into the venv |
| fix | ba2b28cee183d40a45c066b1d821b632f4b580e3 (2.11.17.dev6+gba2b28c) | `pip install --no-deps --no-build-isolation --target $LMX/site-fix .` |
| roevt | scheme1-roe-vt head (2.11.17.dev7+gfb1baeb) | `pip install --no-deps --no-build-isolation --target $LMX/site-roevt .` |

Each is a CPU Release build following the README:
`cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON && cmake --build build --parallel 5`
(CUDA off). gcc 11.5.0 (Red Hat 11.5.0-14), Python 3.11.13, torch 2.10.0 (+cu128 wheel, run on CPU),
kintera 2.6.6, pyharp 2.7.6, numpy 2.4.6. fp64, one thread per run.
Unit tests on the roevt build: `ctest -R "vic_partial_tangential|backward_substitution|implicit_face_work_jacobian|implicit_cfl"` -> 4/4 pass
(after the test change described under check 1; the unchanged reduced-full-matrix test fails on the variant by design).

## Deck and driver

`run_check.py` is `scheme1_donor_momentum/run_arm.py` (Xinyue's driver at
9825aba9, itself the `lmars-energy-flux-explicit` deck): 100 K isothermal dry
H2-He, g 8.87, 450 km / 100 levels, reflecting walls, x2 periodic, lmars, weno5,
rk3, cfl 0.3, uniform-random 1e-6 K isobaric noise (seed 20261026), 3 days,
diagnostics every 600 s. Additions:

- `--shear`: u(z) = u0 z / 450 km at cell centres (2 to 398 m/s for u0 = 400).
- `--moist`: dry + vapor + cloud with the `vapor <=> cloud` nucleation reaction
  of `tests/test_tracer_dry_convention.yaml` (32 km / 32 levels, g 10,
  isothermal 320 K, 1e5 Pa, vapour 0.02 and cloud 0.005 of the total mass,
  continuous hydrostatic profile, not discretely balanced).
- `--prof-days`: horizontal-mean m_t (= m2) and u profiles at 1, 2, 3 d (`*.prof.csv`).
- Velocities and m_t use the total density (dry + every species).

Arms (`jobs.txt`, `jobs2.txt`, `jobs3.txt`, `jobs_roevt.txt`): box = 64 columns;
col / shear / moist = 4 identical periodic columns (`--uniform-x2`), so the column
stays exactly 1-D while dt keeps the horizontal limit; `n2` = 1e-2 K noise.

## Exact commands

    LMX=/data00/ai_workspace/worker1/lmx   # build host paths
    ./run.sh fix jobs.txt 3; ./run.sh base jobs.txt 3
    ./run.sh roevt jobs_roevt.txt 2; ./run.sh roevt jobs3.txt 2
    ./run.sh fix jobs2.txt 3; ./run.sh base jobs2.txt 3
    $LMX/venv/bin/python analyze.py base fix roevt > tables.md

`fix/moist_u400_s0` (explicit reference, dt 0.23 s, about 1.1e6 steps) was stopped
at 0.46 d (rc 137) and dropped from the job list. It is not needed for any verdict.

## Results (every number is in `tables.md`; line numbers are at ba2b28ce)

### Check 1/5: |A| fold under shear

Scheme 1 vs 9 gap: max_k |<u>_1 - <u>_9| / u0 at 1, 2 and 3 d (it is set in the first hours, then constant).

| arm | base | fix | roevt |
|---|---|---|---|
| uniform column, 1e-6 K | 2.48e-9 | 0 | 0 |
| shear column, 1e-6 K | 2.41e-9 | 1.84e-11 | 1.84e-11 |
| uniform column, 1e-2 K | 2.65e-5 | 2.3-5.2e-14 | 2.6-4.7e-14 |
| shear column, 1e-2 K | 2.57e-5 | 1.99e-7 | 1.98e-7 |

The m_t-profile gaps follow the same pattern: shear 8.46e-10 -> 8.54e-11 -> 8.54e-11, uniform 8.63e-10 -> 0 -> 0.
In every column rms w is at round-off (about 1e-13 m/s) in both schemes, so the
rms-w gap there is noise. The 2-D shear box (u(z) 0 -> 400, dx 100 km) gives
the rms-w comparison:

| box, dx 100 km | s1 rate (1/d) | s9 rate | rms w gap s1 vs s9, max over run (1, 2, 3 d) |
|---|---|---|---|
| uniform, base | +31.44 | -0.248 | 4.4e6 |
| uniform, fix | -0.248 | -0.248 | 1.41e-6 (7.9e-8, 5.3e-8, 8.1e-7) |
| uniform, roevt | -0.248 | -0.248 | 2.36e-6 |
| shear, base | +1.107 | -0.188 | 1.7e4 |
| shear, fix | -0.188 | -0.188 | 1.05e-3 (1.1e-4, 2.0e-4, 2.1e-4) |
| shear, roevt | -0.188 | -0.188 | 1.05e-3 (1.0e-4, 2.0e-4, 2.2e-4) |

The shear gap is clearly larger than the uniform one, so the variant was
implemented (`scheme1-roe-vt`: Jacobian part with the cell's v_t, |A| part and
its face-work rows with the face's Roe-averaged v_t). It changes none of the
gaps (to 3 digits). Final-state m2 differs from ba2b28ce by at most 1.1e-12
(shear box) and 3.8e-11 (1e-2 K shear column), relative, against s1-vs-s9 gaps
of 7.4e-10 and 8.5e-7. Variant regression: u = 400 boxes at dx 50/100/200 km decay at
-0.333/-0.248/-0.113 per day (fix: the same), and the uniform column holds at 5.09e-13 (fix: 5.09e-13).
Scheme 9 is bitwise identical at base, fix and roevt (equal `state_sha256`).
Hypothesis, untested: under shear the residual comes from m_t staying
explicit in scheme 1 (q_t on the right-hand side plus the post-solve donor
transfer at the upwind cell's v_t), not from the |A| v_t choice.

### Check 2: du_rho for a moist deck

`fold_cell` takes `drho = du[IDN] + sum_{n<ny} du[ICY+n]`
(src/implicit/vic_assemble_partial_impl.h:201-202), with `ny = nhydro - ICY`, i.e. every
species row (src/implicit/implicit_dispatch.cpp:32). In the ideal-moist EOS,
cons[IDN] is the dry density and cons[ICY+n] the partial density of each vapour
and condensate (src/eos/ideal_moist.cpp:141-146). The primitive density is their
sum (172-174), and v_t = m_t / total density (182). So du_rho is the TOTAL
mass increment, the same sum the solve's mass row uses
(src/implicit/forward_sweep_impl.h:43-45, 99-101). Moist uniform-wind column,
u = 400, max|u - u0|/u0 over 3 d: base 5.35e-6, fix 5.44e-13, roevt 5.44e-13,
scheme 9 5.41e-13.

### Check 3: x1 seams

Not applicable. Before every implicit solve, `HydroImpl::_apply_implicit_correction`
requires both x1 faces of the block to be physical ("implicit scheme requires nb1 = 1",
src/hydro/hydro.cpp:426-431). Its only caller is src/hydro/hydro_forward.cpp:1167,
and it is the only path to `forward_masked` (src/hydro/hydro.cpp:451). The CPU and
CUDA dispatchers hard-code `first_block = last_block = true`
(src/implicit/implicit_dispatch.cpp:33-34, 92-93, 148-149;
src/implicit/implicit_dispatch.cu:43-44, 101-102), so the mirror is applied only
at real walls. x1 splitting (nb1 > 1, src/layout/layout.cpp:236) exists only for explicit runs.

### Check 4: gravity work in the E row

The E row has face gravity work: Am2/Ap2 row 0 plus the m1 column
(src/implicit/vic_assemble_partial_impl.h:149-166). The fold gives the m_t columns
the same rows (183-191), and rhs_corr = tb * q_t includes them (199-209), so the
q_t mass flux brings its gravity work into E. Relative E+PE drift over 3 d,
end [max]:

| arm | base s1 | fix s1 | roevt s1 | s9 (all builds) |
|---|---|---|---|---|
| box dx 100, u 400 | -1.48e-8 [1.96e-8] | -5.49e-13 | -5.49e-13 | -5.49e-13 |
| shear box dx 100 | -2.07e-11 | -5.59e-13 | -5.60e-13 | -5.60e-13 |
| shear column | -2.77e-13 | -2.76e-13 | -2.78e-13 | -2.77e-13 |
| box dx 50 / 200 | -1.11e-12 / blew up (-4.2e-6) | -1.10e-12 / -2.73e-13 | -1.10e-12 / -2.73e-13 | n/a |

The moist column drifts by 1.95e-10 in every scheme, scheme 9 included. Hypothesis,
untested: the E + rho g z diagnostic does not match the moist EOS energy
(reference energies, phase change), so this isn't a scheme-1 effect.

Base box dx 200 km stopped at 0.135 d with more than 50 redos (rate 375/d in the band fit).
Base dx 50 km grows at +1.77/d, as in the earlier rerun.
