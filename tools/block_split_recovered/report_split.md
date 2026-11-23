# nb2 vs nb1 split dependence: main dcf5a5a1 vs #365 0e240e15 (1 rank)

Pin: git ls-remote at 2026-11-15T06:32:56Z: refs/heads/main dcf5a5a12e644b67232adb718bb6166b2ef0c0ae (as expected),
refs/pull/365/head 0e240e15e12f09669f07068b75fd3ef76db631a4 (as expected). Main's tree 20b6730d321357c9e03e40ae3f3a536d746d486c
is identical to c8a0a309's, the parent of 0e240e15, so #365 = main + the closure commit only.

Build of main: fresh checkout, own build dir, CPU Release Gloo-only per README (same flags as the 0e240e15 build:
-DCMAKE_BUILD_TYPE=Release -DNETCDF=ON -DCUDA=OFF -DUCX=OFF -DBUILD_TESTS=OFF, --parallel 2, pip --target): 981 s, 0 compiler
warnings, snapy 0.0.1.dev483+gdcf5a5a12 (the fresh fetch has no tags, hence the version string), torch 2.10.0+cu128.
0e240e15 build reused as is (snapy 2.13.25.dev2+g0e240e1).

Cases (same driver and the same Python helpers, from the 0e240e15 test file, for both builds): f, g, h at Omega = wind rate
("slow") and 7e-5 ("deck"), 20 steps; W2 at Omega and 4 Omega, 200 steps; interp_order 4; 1 rank; nb 1 (6 blocks) vs nb 2
(2x2 per panel). Metric: max over variables of max|u_nb2 - u_nb1| / max|u_nb1|; rows of hydro_u 0 rho, 1 m1 (vertical), 2 m2, 3 m3, 4 E.

## max rel diff nb2 vs nb1 (dt sequences equal? Y/N)
| build / config | f slow | f deck | g slow | g deck | h slow | h deck | W2 1Ω | W2 4Ω |
|---|---|---|---|---|---|---|---|---|
| main, full (no key) | 0 Y | 0 Y | 8.1e-8 N | 6.8e-8 N | 3.5e-12 N | 0 Y | 0 Y | 0 Y |
| main, transport | 0 Y | 0 Y | 7.1e-8 N | 8.0e-8 N | 3.8e-12 N | 1.8e-12 N | 0 Y | 0 Y |
| 0e240e15, full, closure false | 0 Y | 0 Y | 8.1e-8 N | 6.8e-8 N | 3.5e-12 N | 0 Y | 0 Y | 0 Y |
| 0e240e15, transport | 0 Y | 0 Y | 7.1e-8 N | 8.0e-8 N | 3.8e-12 N | 1.8e-12 N | 0 Y | 0 Y |
| 0e240e15, full, closure true | 0 Y | 0 Y | 6.8e-8 N | 9.9e-8 N | 3.2e-12 N | 2.7e-12 N | 0 Y | 1.2e-15 Y |

Per variable, every nonzero case is dominated by row 1 (m1, vertical momentum): g ~7-10e-8 in m1 vs 1.5-2.5e-11 in m2/m3
and ~1e-12 in rho/E; h ~2-4e-12 in m1 vs 1e-14 to 1e-13 in m2/m3 and ~3e-15 in rho/E (full lists in runs/cmp_*.txt).
Cross-build: main full (no key) vs 0e240e15 full with closure false, and main transport vs 0e240e15 transport, are bitwise
equal (fields and dt) in 16/16 (case, nb) runs each.
Closure true reproduces the earlier numbers (g 6.8e-8 / 9.9e-8, h 3.2e-12 / 2.7e-12).

## Verdict
The split dependence is on main and does not come from the closure: main shows it with full and with transport, at the same
size as the closure runs (main full = 0e240e15 closure-off bitwise). The closure only shifts it within the same range
(g 6.8e-8 to 9.9e-8 vs 6.8e-8 to 8.1e-8 without). It is largest on arm g (traditional Coriolis, 8 m shell, implicit 9):
8.1e-8 (full, slow) and 8.0e-8 (transport, deck), all in the vertical momentum, with the dt sequence changing; then h
(implicit 9 + w, 1.8-3.8e-12). The explicit arm f and shallow-water W2 are bitwise split-independent.
Hypothesis (not tested): the vertically implicit path (implicit-scheme 9), or the dt it feeds into, depends on the block split.
Only the implicit arms show it, and it sits in m1.

Wall (s, nb1/nb2, 1 thread, machine at load ~15 from another job): main 3-D 0.7-1.1 / 1.7-2.3, W2 3.3-4.4 / 7.6-10.4;
0e240e15 3-D 0.8-1.4 / 2.3-4.2, W2 3.8-5.7 / 11.2-18.1.
