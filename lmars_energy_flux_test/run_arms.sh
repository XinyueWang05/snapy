#!/bin/bash
# Run arms in parallel, longest first.
# usage: run_arms.sh <runs dir> <jobs> <set> [extra run_arm.py args]
#   set = gate   : u = 0, NO noise, {lmars,hllc} x dx {5,50,400} km (balance check)
#   set = matrix : {lmars,hllc} x u {0,50,250,400} x dx {5,50,400} km, 1e-6 K noise
# DXS="50 400" overrides the dx list (km), US="5000 3000 1000 0" the matrix u list (m/s);
# arms whose run dir already exists are skipped (resume a partial launch)
# needs: PYTHON (interpreter with torch/kintera), SNAPY_PY (dir holding the built
# snapy package, put first on PYTHONPATH)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=$1; JOBS=$2; SET=$3; shift 3
mkdir -p "$RUNS"
export PYTHONPATH=$SNAPY_PY OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
list=()
for dx in ${DXS:-5 50 400}; do
  if [ "$SET" = gate ]; then
    for s in lmars hllc; do list+=("$s 0 $dx --noise 0"); done
  else
    for u in ${US:-400 250 50 0}; do for s in lmars hllc; do list+=("$s $u $dx"); done; done
  fi
done
printf '%s\n' "${list[@]}" | xargs -P "$JOBS" -L 1 bash -c '
  s=$0; u=$1; dx=$2; shift 2
  tag=${s}_u${u}_dx${dx}
  d='"$RUNS"'/$tag
  if [ -e $d ]; then echo "$tag exists, skipped"; exit 0; fi
  mkdir -p $d
  cd $d && '"$PYTHON"' '"$HERE"'/run_arm.py --solver $s --u $u --dx ${dx}e3 --out . "$@" '"$*"' > run.log 2>&1
  echo "$tag exit $? $(date -u +%FT%TZ)"'
