# Scheme-1 donor-momentum prototype (implicit-scheme 1, partial VIC)

Change (commit before this one): `src/implicit/vic_redistribute_impl.h`
`vic_tangential_momentum_column`, called for N == 3 after
`vic_constituent_column` in `implicit_dispatch.cpp` and `implicit_dispatch.cu`.
Each face moves `q * v_t(donor)` of both tangential momenta, where q is the
total clamped face mass transfer of the implicit correction. Unit test:
`tests/test_backward_substitution.cpp`
`vic_redistribution.tangential_momentum_follows_the_face_mass`.

Deck and driver: `run_arm.py` is the box driver of the sibling
`lmars-energy-flux-explicit` study (100 K dry H2-He, 450 km / 100 levels,
64 periodic columns, lmars, weno5, rk3, cfl 0.3, 1e-6 K noise, seed 20261026,
3 days, diagnostics every 600 s), plus the perturbation KE and the sha256 of
the final conserved state. base = 85a3ce1 (a09cc256 core), fix = this
branch; both are CPU Release builds, torch 2.10.0, kintera 2.6.6, fp64, one
thread per run. `run.sh <base|fix> <jobs> <npar>` (paths inside are those of
the build host), then `analyze.py runs results.md summary.png`.

The baseline rerun reproduces the explicit study's `box_dx100_u400_s1.csv`
byte for byte (31.44/day). Tables: `results.md`.
