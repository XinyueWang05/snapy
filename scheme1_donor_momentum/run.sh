#!/bin/bash
# usage: run.sh <tree: base|fix> <jobs file> <npar>
# each jobs line: <name> <run_arm.py args>; outputs runs/<tree>/<name>.*
here=$(cd "$(dirname "$0")" && pwd)
tree=$1; jobs=$2; npar=$3
case $tree in base) src=/data00/ai_workspace/worker1/s1/base;; fix) src=/data00/ai_workspace/worker1/s1/src;; esac
export PYTHONPATH=$src/pyinst OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
       MPLCONFIGDIR=/data00/ai_workspace/worker1/s1/.mpl
PY=/data00/ai_workspace/worker3/r321/venv/bin/python
mkdir -p $here/runs/$tree
one() {
  name=$1; shift
  out=$here/runs/$tree/$name
  [ -f $out.json ] && { echo "skip $tree/$name"; return; }
  (cd $here/runs/$tree && $PY $here/run_arm.py --out $out "$@" > $out.log 2>&1)
  echo "$tree/$name exit $? $(date -u +%FT%TZ)"
}
export -f one; export here tree PY
xargs -P $npar -L 1 bash -c 'one "$@"' _ < $jobs
