#!/bin/bash
# Prints one arm per line: <name> <run_arm.py args>. Order = longest first.
for dx in 75 100 125 150; do for u in 110 150 250 400; do
  echo "x_dx${dx}_u${u}_s0 --scheme 0 --dx ${dx}e3 --u0 $u"
done; done
echo "x_dx100_u0_s0_nonoise --scheme 0 --dx 100e3 --u0 0 --noise 0"
for s in 9 1; do echo "box_dx100_u400_s$s --scheme $s --dx 100e3 --u0 400"; done
for s in 0 1 9; do for u in 0 100 200 260 400; do
  echo "col_u${u}_s$s --scheme $s --dx 100e3 --nx2 4 --uniform-x2 --u0 $u"
done; done
