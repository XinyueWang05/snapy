| arm | tree | status | steps | redos | rate (1/d) | fit | rate 0.5-3 d (1/d) | rms w end (m/s) | rms(u-u0) end | drift rho | drift m2 | m3 | drift E (no PE) | drift E+rho g z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| box_dx50_u400_s1 | base | finished | 19740 | 0 | +1.77 | run 0.50-3.00 d | +1.77 | 1.42e-05 | 0.00123 | 1.11e-12 | 1.14e-12 | 0 | 7.18e-11 | 1.11e-12 |
| box_dx50_u400_s1 | fix | finished | 19741 | 0 | +2.92 | band 1.59-3.00 d | +2.85 | 0.0067 | 0.0346 | 1.11e-12 | 1.14e-12 | 0 | 7.63e-11 | 1.12e-12 |
| box_dx100_u400_s1 | base | finished | 11543 | 13 | +31.44 | band 0.17-0.38 d | +0.15 | 0.419 | 19.9 | 6.46e-13 | 6.67e-13 | 0 | 7.02e-05 | 1.96e-08 |
| box_dx100_u400_s1 | fix | finished | 9870 | 0 | +12.71 | band 2.96-3.00 d | +1.69 | 0.000181 | 0.00149 | 5.53e-13 | 5.70e-13 | 0 | 7.11e-11 | 5.49e-13 |
| box_dx100_u400_s9 | base | finished | 9870 | 0 | -0.25 | run 0.50-3.00 d | -0.25 | 4.16e-07 | 1.21e-05 | 5.54e-13 | 5.70e-13 | 0 | 7.11e-11 | 5.49e-13 |
| box_dx100_u400_s9 | fix | finished | 9870 | 0 | -0.25 | run 0.50-3.00 d | -0.25 | 4.16e-07 | 1.21e-05 | 5.54e-13 | 5.70e-13 | 0 | 7.11e-11 | 5.49e-13 |
| box_dx200_u400_s1 | base | blew_up(redo>50) | 295 | 51 | +375.07 | band 0.03-0.04 d | +nan | 4.43 | 62.7 | 1.69e-14 | 1.68e-14 | 0 | 2.29e-03 | 4.84e-06 |
| box_dx200_u400_s1 | fix | finished | 4935 | 0 | +0.77 | run 0.50-3.00 d | +0.77 | 5.05e-06 | 9.05e-05 | 2.77e-13 | 2.85e-13 | 0 | 7.09e-11 | 2.73e-13 |
| box_dx400_u400_s1 | base | blew_up(redo>50) | 72 | 51 | +nan | rms w = 0 | +nan | 7.01 | 90.4 | 4.66e-15 | 3.98e-15 | 0 | 7.67e-03 | 4.74e-05 |
| box_dx400_u400_s1 | fix | finished | 2468 | 0 | +0.22 | run 0.50-3.00 d | +0.22 | 5.08e-07 | 2.08e-05 | 1.38e-13 | 1.42e-13 | 0 | 7.11e-11 | 1.35e-13 |
| box_dx100_u0_s1 | base | finished | 6414 | 0 | -0.29 | run 0.50-3.00 d | -0.29 | 6.45e-08 | 3.72e-06 | 3.60e-13 | 5.04e-25 (/S_rho c_s) | 0 | 7.66e-11 | 3.64e-13 |
| box_dx100_u0_s1 | fix | finished | 6414 | 0 | -0.29 | run 0.50-3.00 d | -0.29 | 6.45e-08 | 3.72e-06 | 3.60e-13 | 1.07e-24 (/S_rho c_s) | 0 | 7.66e-11 | 3.63e-13 |
| box_dx100_u0_s1_nonoise | base | finished | 6414 | 0 | -0.01 | run 0.50-3.00 d | -0.01 | 1.13e-13 | 0 | 1.36e-13 | 0 | 0 | 9.94e-13 | 2.03e-13 |
| box_dx100_u0_s1_nonoise | fix | finished | 6414 | 0 | -0.01 | run 0.50-3.00 d | -0.01 | 1.13e-13 | 0 | 1.36e-13 | 0 | 0 | 9.94e-13 | 2.03e-13 |

Bitwise comparison base vs fix (sha256 of the final conserved state array and of the full-precision diagnostic CSV):

| arm | final state identical | csv identical | max abs diff final state | state sha256 (base) |
|---|---|---|---|---|
| box_dx50_u400_s1 | False | False | 117 | 26a910202e12d9a8 |
| box_dx100_u400_s1 | False | False | 2.78e+03 | f93f61eae5d4274d |
| box_dx100_u400_s9 | True | True | 0 | 25a0ccbf207d706b |
| box_dx200_u400_s1 | False | False | 9.49e+04 | 69620ea6c3dcab0e |
| box_dx400_u400_s1 | False | False | 4.91e+05 | e3e81e547a06df0c |
| box_dx100_u0_s1 | False | False | 3.53e-08 | c213d6f2b2d549c1 |
| box_dx100_u0_s1_nonoise | True | True | 0 | a7ebceb5fe50acfe |
