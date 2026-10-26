#!/bin/bash
# Detached launcher: runs every arm of jobs.txt, NPAR at a time, skipping arms
# whose .json summary exists (so a relaunch resumes). State: runs/<name>.state
here=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
NPAR=${NPAR:-5}
mkdir -p "$here/runs"
one() {
  name=$1; shift
  out="$here/runs/$name"
  [ -f "$out.json" ] && { echo "skip $name"; return; }
  echo "running pid=$$ start=$(date -u +%FT%TZ)" > "$out.state"
  OMP_NUM_THREADS=1 "$PY" "$here/run_arm.py" --out "$out" "$@" > "$out.log" 2>&1
  rc=$?
  echo "exit rc=$rc end=$(date -u +%FT%TZ)" >> "$out.state"
}
export -f one; export here PY
echo "launcher pid=$$ start=$(date -u +%FT%TZ)" > "$here/runs/LAUNCHER.state"
xargs -P "$NPAR" -L 1 bash -c 'one "$@"' _ < "$here/jobs.txt"
echo "launcher done $(date -u +%FT%TZ)" >> "$here/runs/LAUNCHER.state"
