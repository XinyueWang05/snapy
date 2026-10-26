#!/bin/bash
# Extra diagnostic (not in the original brief): the u = 400 m/s, dx = 5 km arm
# with the vertical solve EXPLICIT (integration/implicit-scheme 0) for 1 day,
# both solvers. Same deck, noise and seed otherwise. dz = 4.5 km < dx, so the
# explicit step is still set by the horizontal CFL and costs the same.
# usage: run_diag_scheme0.sh <runs dir>    (needs PYTHON, SNAPY_PY as run_arms.sh)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SNAPY_PY OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in lmars hllc; do
  d=$1/${s}_u400_dx5; mkdir -p $d
  (cd $d && $PYTHON $HERE/run_arm.py --solver $s --u 400 --dx 5e3 --out . --days 1 --scheme 0 > run.log 2>&1; echo "$s scheme0 exit $?") &
done
wait
